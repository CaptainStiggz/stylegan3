import cv2
import numpy as np
import os
import random
import shutil

def read_file(file):
  img = cv2.imread(file)
  return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def write_file(img, file):
  return cv2.imwrite(file, img)

def rand_image(dir):
  return read_file(dir+"/"+random.choice(os.listdir(dir)))

def to_grayscale(img):
  return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def threshold(img, lower = 200, upper = 255):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, lower , upper, cv2.CHAIN_APPROX_NONE)
    return gray

def find_largest_threshold_contour(img, lower = 200, upper = 255):
  gray = threshold(img, lower, upper)
  return find_largest_contour(gray)

def find_largest_contour(image):
  image = image.astype(np.uint8)
  contours,_ = cv2.findContours(
    image,
    cv2.RETR_TREE,
    cv2.CHAIN_APPROX_SIMPLE
  )
  return max(contours, key=cv2.contourArea)

def crop(img, crop):
  x, y, w, h = crop
  return img[y:y+h,x:x+w]

def bbox_contour(contour, pad):
  _x, _y, _w, _h = cv2.boundingRect(contour)
  x, y, w, h = _x, _y, _w, _h
  diff = w - h
  if diff > 0:
    x -= pad
    w += 2 * pad
    y = y - diff // 2 - pad
    h = w
  elif diff < 0:
    y -= pad
    h += 2 * pad
    x = x + diff // 2 - pad
    w = h
  if x < 0 or y < 0:
    print("BBOX TOO CLOSE TO EDGE")
    return None
  if w - h > 2:
    print("BBOX IS NOT SQUARE!")
    return None
  return x,y,w,h

def saturate(img, satval):
  imghsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype("float32")
  h, s, v = cv2.split(imghsv)
  s = s*satval
  s = np.clip(s,0,255)
  imgmer = cv2.merge([h,s,v])
  return cv2.cvtColor(imgmer.astype("uint8"), cv2.COLOR_HSV2RGB)

def contrast(img, contrast):
  clip, k = contrast
  lab= cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
  l, a, b = cv2.split(lab)
  clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(k,k))
  cl = clahe.apply(l)
  limg = cv2.merge((cl,a,b))
  return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def process_dir(dir, func, limit = 0):
  out = dir+"_out"
  if os.path.exists(out):
    shutil.rmtree(out)
  os.makedirs(out)

  i = 0
  for file in os.listdir(dir):
    if file.endswith(".jpg"):
      func(dir+"/"+file, out+"/"+file)
      i += 1
      if limit > 0 and i > limit:
        break

def print_progress_bar(current, total, barLength = 20):
    percent = float(current) * 100 / total
    arrow   = '-' * int(percent/100 * barLength - 1) + '>'
    spaces  = ' ' * (barLength - len(arrow))

    print('Progress: [%s%s] %d %%' % (arrow, spaces, percent), end='\r')

def get_file_count(dir, filetype = ".jpg"):
  i = 0
  for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
      if name.endswith(filetype):
        i += 1
  return i

def run_on_dir(dir, func, filetype = ".jpg"):
  i = 0
  cnt = get_file_count(dir, filetype)
  print(f'Processing {cnt} files...')
  for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
      if name.endswith(filetype):
        func(f'{root}/{name}')
        print_progress_bar(i, cnt)
        i += 1
        
        

# def contrast(img):
#   img_grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#   return img_grey.std()