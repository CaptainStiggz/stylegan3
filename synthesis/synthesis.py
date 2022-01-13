import dnnlib
import legacy
import torch
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import scipy.interpolate
import math
import random
import preprocessing.image_utils as itools
import cv2
from tqdm import tqdm

def load_generator(path, device):
  print('Loading networks from "%s"...' % path)
  with dnnlib.util.open_url(path) as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore
    print("Network loaded.")
    return G

def make_transform(translate, angle):
  m = np.eye(3)
  s = np.sin(angle/360.0*np.pi*2)
  c = np.cos(angle/360.0*np.pi*2)
  m[0][0] = c
  m[0][1] = s
  m[0][2] = translate[0]
  m[1][0] = -s
  m[1][1] = c
  m[1][2] = translate[1]
  return m

def synthesize_image(z, G, device, translate=(0,0), rotate=0, noise_mode = "const", truncation_psi = 1):
  label = torch.zeros([1, G.c_dim], device=device)
  # Construct an inverse rotation/translation matrix and pass to the generator.  The
  # generator expects this matrix as an inverse to avoid potentially failing numerical
  # operations in the network.
  if hasattr(G.synthesis, 'input'):
    m = make_transform(translate, rotate)
    m = np.linalg.inv(m)
    G.synthesis.input.transform.copy_(torch.from_numpy(m))

  img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
  print(img.shape)
  img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
  return img[0].cpu().numpy()

# 9 
# 13
# 16
# 20
# 21
# 25
# 27
# 28
# 31
# 34
# 37
# 38
# 39
# 43
# 49
# 50
# 400

def synthesize_rand_image(seed, G, device):
  z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
  return synthesize_image(z)

def find_smallest_bbox(bbox):
  x, y, w, h = bbox
  diff = w - h
  if diff > 0:
    y = y - diff // 2
    h = w
  elif diff < 0:
    x = x + diff // 2
    w = h
  if w != h:
    raise ("BBOX IS NOT SQUARE!")
  return x, y, w, h

def crop_to_smallest_bbox(img):
#     contrast = itools.contrast(img, (3, 8))
  sat = itools.saturate(img, 5)
  gray = itools.threshold(sat, 210)
  contour = itools.find_largest_contour(gray)

  if contour is None:
    raise "NO CONTOUR"

  # get the bounding box
  bbox = cv2.boundingRect(contour)

  # delete if there is no bounding box
  if bbox is None:
    raise "NO BBOX FOUND"

  crop = find_smallest_bbox(bbox)
  cropped = itools.crop(img, crop)
  return cropped

def synthesize_rand_interp(seeds, G, device, w_frames = 20):
  print("interpolating: ", seeds)
  num_keyframes = len(seeds) // 1
  wraps = 1
  kind= "cubic"
  imgs = []
  
  zs = torch.from_numpy(np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in seeds])).to(device)
  ws = G.mapping(z=zs, c=None, truncation_psi=1)
  _ = G.synthesis(ws[:1]) # warm up
  print("ws.shape original", ws.shape)
  print("ws[:1].shape", ws.shape[:1])
  ws = ws.reshape(1, 1, num_keyframes, *ws.shape[1:])
  print("ws.shape", ws.shape)
  
  # wraps = 0
  # ws.shape torch.Size([1, 1, 2, 16, 512])
  # x.shape, y.shape (2,) (2, 16, 512)
  
  # wrapping back to the original
#     x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
#     y = np.tile(ws[0][0].cpu().numpy(), [wraps * 2 + 1, 1, 1])
  
  x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
  y = np.tile(ws[0][0].cpu().numpy(), [wraps * 2 + 1, 1, 1])

#     x = np.arange(0, 2)
#     y = np.tile(ws[0][0].cpu().numpy(), [1, 1, 1])
  print("x.shape, y.shape", x.shape, y.shape)
  interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
  
  for frame_idx in tqdm(range(num_keyframes * w_frames)):
    w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
    img = G.synthesis(ws=w.unsqueeze(0), noise_mode='const')[0]
    img = img.reshape(1, 3, 1024, 1024)
    img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
#         img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.permute(0, 2, 3, 1)
    img = img[0].cpu().numpy() # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
    imgs.append(img)
  
  return imgs

def show_imgs(imgs):
  print("images", len(imgs))
  cols = min(len(imgs), 6)
  rows = max(math.ceil(len(imgs) / cols), 1)
  _, axs = plt.subplots(rows, cols, figsize=(12, 12))
  axs = axs.flatten()
  for img, ax in zip(imgs, axs):
    ax.imshow(img)
  plt.show()

def create_kaleidoscope(imgs):
  # crop to bbox
  cropped_imgs = []
  for img in imgs:
    cropped = crop_to_smallest_bbox(img)
    cropped_imgs.append(cropped)
  # normalize size
  size = None
  for img in cropped_imgs:
    if size is None or img.shape[0] < size:
      size = img.shape[0]
  resized_imgs = []
  for img in cropped_imgs:
    pil = PIL.Image.fromarray(img, 'RGB')
    resized = pil.resize((size, size), resample=PIL.Image.BICUBIC)
    resized_imgs.append(np.array(resized))
  
  count = len(imgs) 
  dims = count * 2 - 1
  r = dims // 2
  # construct output
  out = np.zeros((dims * size, dims * size, 3)).astype(np.uint8)

  for row in range(dims):
    for col in range(dims):
      x = col - r
      y = r - row
      l = round(math.sqrt(x ** 2 + y ** 2))
      out[col * size: (col+1)* size, row * size: (row+1)* size] = resized_imgs[min(l, count - 1)]
  
  print("out.shape", out.shape, out.dtype)
  return out

def synthesize_kaleidoscope(G, device, num_seeds = 4, w_frames = 20):
  seeds = [random.randint(1, 10000) for _ in range(num_seeds)]
  print(seeds)
  imgs = synthesize_rand_interp(seeds, G, device, w_frames)
  return create_kaleidoscope(imgs)