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
import imageio
import cv2
import os
import boto3
import time

from tqdm import tqdm

from typing import List, Optional, Tuple, Union


def load_generator(path, device):
    print('Loading networks from "%s"...' % path)
    with dnnlib.util.open_url(path) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device)  # type: ignore
        print("Network loaded.")
        return G


def make_transform(translate, angle):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


def synthesize_image(
    z, G, device, translate=(0, 0), rotate=0, noise_mode="const", truncation_psi=1
):
    label = torch.zeros([1, G.c_dim], device=device)
    # Construct an inverse rotation/translation matrix and pass to the generator.  The
    # generator expects this matrix as an inverse to avoid potentially failing numerical
    # operations in the network.
    if hasattr(G.synthesis, "input"):
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


def get_seed():
    return random.randint(1, 1000000000)


def synthesize_rand_image(seed, G, device):
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    return synthesize_image(z, G, device)


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


def synthesize_rand_interp(seeds, G, device, w_frames=20, wrap=False):
    print("interpolating: ", seeds)
    num_keyframes = len(seeds) // 1
    wraps = 1
    kind = "cubic"
    imgs = []

    zs = torch.from_numpy(
        np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in seeds])
    ).to(device)
    ws = G.mapping(z=zs, c=None, truncation_psi=1)
    _ = G.synthesis(ws[:1])  # warm up
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
        img = G.synthesis(ws=w.unsqueeze(0), noise_mode="const")[0]
        img = img.reshape(1, 3, 1024, 1024)
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        #         img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        img = img.permute(0, 2, 3, 1)
        img = img[0].cpu().numpy()  # PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
        imgs.append(img)

    if not wrap:
        imgs = imgs[:w_frames]
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


def crop_and_normalize(imgs):
    # crop to bbox
    cropped_imgs = []
    for img in imgs:
        cropped = crop_to_smallest_bbox(img)
        cropped_imgs.append(cropped)
    size = None
    for img in cropped_imgs:
        if size is None or img.shape[0] < size:
            size = img.shape[0]
    resized_imgs = []
    for img in cropped_imgs:
        pil = PIL.Image.fromarray(img, "RGB")
        resized = pil.resize((size, size), resample=PIL.Image.BICUBIC)
        resized_imgs.append(np.array(resized))
    return resized_imgs


def create_kaleidoscope(imgs):
    resized_imgs = crop_and_normalize(imgs)

    count = len(imgs)
    dims = count * 2 - 1
    r = dims // 2
    # construct output
    out = np.zeros((dims * size, dims * size, 3)).astype(np.uint8)

    for row in range(dims):
        for col in range(dims):
            x = col - r
            y = r - row
            l = round(math.sqrt(x**2 + y**2))
            out[
                col * size : (col + 1) * size, row * size : (row + 1) * size
            ] = resized_imgs[min(l, count - 1)]

    print("out.shape", out.shape, out.dtype)
    return out


def synthesize_kaleidoscope(G, device, seeds, w_frames=20):
    print(seeds)
    imgs = synthesize_rand_interp(seeds, G, device, w_frames)
    return create_kaleidoscope(imgs)


def create_kaleidoscopes(G, device):
    for _ in range(100):
        try:
            num_seeds = random.randint(2, 20)
            w_frames = random.randint(2, 20)
            seeds = [get_seed() for _ in range(num_seeds)]
            seeds_str = ",".join(str(seed) for seed in seeds)
            print(
                f"creating kaleidoscope with seeds: {num_seeds}, w_frames: {w_frames}"
            )
            k = synth.synthesize_kaleidoscope(G, device, num_seeds, w_frames)
            filename = f"ns{num_seeds}-wf{w_frames}-[{seeds_str}].jpg"
            PIL.Image.fromarray(k, "RGB").save(f"../kaleidoscopes/{filename}")
        except:
            print("Failed...continuing.")


def generate_samples(count, outdir, G, device):
    seeds = [get_seed() for _ in range(count)]
    count = 0
    for seed in seeds:
        img = synthesize_rand_image(seed, G, device)
        filename = f"sample-{seed}.jpg"
        PIL.Image.fromarray(img, "RGB").save(f"{outdir}/{filename}")
        count += 1


def generate_video(dims, duration, outdir, G, device, upload=False):
    # Formula for video length is (# seeds / (W * H)) * 3 = length in seconds
    w, h = dims
    num_seeds = (duration * w * h) / 2
    print("Seed count: ", num_seeds)
    seeds = [get_seed() for _ in range(int(num_seeds))]
    seeds_str = ",".join(str(seed) for seed in seeds)
    name = time.time()
    filename = f"{outdir}/{w}x{h}-{name}.mp4"
    seeds = fix_seeds(seeds, G, device)
    gen_interp_video(G, filename, seeds, device=device, grid_dims=dims)
    if upload:
        upload_file(filename, "")


def generate_videos(count, outdir, G, device, size=(1, 1), duration=6, upload=False):
    for _ in range(count):
        generate_video(size, duration, outdir, G, device, upload)


# fix seeds by removing any "funky" images with bad edges
def fix_seeds(seeds, G, device):
    return [fix_seed(seed, G, device) for seed in seeds]


def fix_seed(seed, G, device):
    img = synthesize_rand_image(seed, G, device)
    ok = avg_edge_pixels_above_threshold(img)
    if ok:
        return seed
    else:
        print(f"Fixing seed: {seed}")
        return fix_seed(get_seed(), G, device)


# TODO: put this in a different place


def upload_file(file_name, bucket, object_name=None):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(file_name)

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        print("upload error", e)
        return False
    return True


# hallway stuff
# ----------------------------------------------------------------------------


def stitch(imgs, mode="horizontal"):
    height, width, _ = imgs[0].shape
    shape = (height, width * len(imgs), 3)
    if mode == "vertical":
        shape = (height * len(imgs), width, 3)
    out = np.zeros(shape, dtype="uint8")
    for i in range(len(imgs)):
        if mode == "vertical":
            s = i * height
            out[s : s + height, :, :] = imgs[i]
        else:
            s = i * width
            out[:, s : s + width, :] = imgs[i]
    return out


def stack(img, mode="n", n=10):
    height, width, _ = img.shape
    if mode == "square":
        n = width // height
    out = np.zeros((n * height, width, 3), dtype="uint8")
    for i in range(n):
        h = i * height
        out[h : h + height, :, :] = img
    return out


def transform_perspective(img, stride=1, dw=0.5, dh=0.5):
    print("img.shape", img.shape, img.dtype)
    height, width, _ = img.shape
    # theta = theta * math.pi / 180
    if dh >= 1 or dh <= 0:
        raise "dh parameter must be between 0 - 1, not inclusive"

    vw = width * dw
    vh = height * dh
    pil = PIL.Image.fromarray(img, "RGB")
    img = np.array(pil.resize((round(vw), height), resample=PIL.Image.BICUBIC))
    print("skewed.shape", img.shape)
    _h, _w, _ = img.shape

    out = np.zeros(img.shape, dtype="uint8")
    for i in range(0, _w, stride):
        r = img[:, i : i + stride, :]  # get the slice
        # h = height - 2 * width * math.sin(theta) * i
        h = height - (i / (_w - 1)) * (height - vh)
        # h = height - math.tan(theta) * i # perspective height of the slice
        pil = PIL.Image.fromarray(r, "RGB")  # resize
        r = np.array(pil.resize((r.shape[1], round(h)), resample=PIL.Image.BICUBIC))

        # pad the slice to get the same height as before
        diff = height - r.shape[0]
        pad = diff // 2
        xtra = 0 if not diff % 2 else 1
        npad = ((pad, pad + xtra), (0, 0), (0, 0))
        b = np.pad(r, pad_width=npad, mode="constant", constant_values=0)

        # apply to the new shape
        out[:, i : i + stride, :] = b

    print("out shape", out.shape, out.dtype)
    return out


def create_hallway(imgs, dw=0.25, dh=0.25):
    left = transform_perspective(imgs, dw=dw, dh=dh)
    right = np.copy(left[..., ::-1, :])
    bottom = np.rot90(left)
    top = np.rot90(right)
    print("left.shape", left.shape)
    print("right.shape", right.shape)
    print("bottom.shape", bottom.shape)
    print("top.shape", top.shape)
    out = np.zeros((left.shape[0], bottom.shape[1], 3), dtype="uint8")
    out[:, : left.shape[1], :] += left
    out[:, -right.shape[1] :, :] += right
    out[: top.shape[0], :, :] += top
    out[-bottom.shape[0] :, :, :] += bottom
    out = np.where(out.any(-1, keepdims=True), out, 255)  # black -> white
    return out


def synthesize_hallway(G, device, seeds, w_frames=10, k=0.25):
    imgs = synthesize_rand_interp(seeds, G, device, w_frames)
    imgs = crop_and_normalize(imgs)
    imgs = stitch(imgs)
    stacked = stack(imgs, mode="square")
    h, w, _ = stacked.shape
    l = math.sqrt(k * (h**2))
    dw = ((h - l) / 2) / w
    dh = l / h
    hallway = create_hallway(stacked, dw=dw, dh=dh)
    return hallway


def generate_hallways(count, outdir, G, device):
    for _ in range(count):
        k = 0.01 + random.random() * 0.25
        frames = 10 + round(random.random() * 10)
        seeds = [get_seed() for _ in range(2)]
        seeds_str = ",".join(str(seed) for seed in seeds)
        filename = f"hallway-k{k}-wf{frames}-[{seeds_str}].jpg"
        img = synthesize_hallway(G, device, seeds, w_frames=frames, k=k)
        PIL.Image.fromarray(img, "RGB").save(f"{outdir}/{filename}")


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# COPIED FROM gen_video.py


def layout_grid(
    img, grid_w=None, grid_h=1, float_to_uint8=True, chw_to_hwc=True, to_numpy=True
):
    batch_size, channels, img_h, img_w = img.shape
    if grid_w is None:
        grid_w = batch_size // grid_h
    assert batch_size == grid_w * grid_h
    if float_to_uint8:
        img = (img * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    img = img.reshape(grid_h, grid_w, channels, img_h, img_w)
    img = img.permute(2, 0, 3, 1, 4)
    img = img.reshape(channels, grid_h * img_h, grid_w * img_w)
    if chw_to_hwc:
        img = img.permute(1, 2, 0)
    if to_numpy:
        img = img.cpu().numpy()
    return img


# ----------------------------------------------------------------------------
# COPIED FROM gen_video.py


def gen_interp_video(
    G,
    mp4: str,
    seeds,
    shuffle_seed=None,
    w_frames=60 * 4,
    kind="cubic",
    grid_dims=(1, 1),
    num_keyframes=None,
    wraps=2,
    psi=1,
    device=torch.device("cuda"),
    **video_kwargs,
):
    grid_w = grid_dims[0]
    grid_h = grid_dims[1]

    if num_keyframes is None:
        if len(seeds) % (grid_w * grid_h) != 0:
            raise ValueError("Number of input seeds must be divisible by grid W*H")
        num_keyframes = len(seeds) // (grid_w * grid_h)

    all_seeds = np.zeros(num_keyframes * grid_h * grid_w, dtype=np.int64)
    for idx in range(num_keyframes * grid_h * grid_w):
        all_seeds[idx] = seeds[idx % len(seeds)]

    if shuffle_seed is not None:
        rng = np.random.RandomState(seed=shuffle_seed)
        rng.shuffle(all_seeds)

    zs = torch.from_numpy(
        np.stack([np.random.RandomState(seed).randn(G.z_dim) for seed in all_seeds])
    ).to(device)
    ws = G.mapping(z=zs, c=None, truncation_psi=psi)
    _ = G.synthesis(ws[:1])  # warm up
    ws = ws.reshape(grid_h, grid_w, num_keyframes, *ws.shape[1:])

    # Interpolation.
    grid = []
    for yi in range(grid_h):
        row = []
        for xi in range(grid_w):
            x = np.arange(-num_keyframes * wraps, num_keyframes * (wraps + 1))
            y = np.tile(ws[yi][xi].cpu().numpy(), [wraps * 2 + 1, 1, 1])
            interp = scipy.interpolate.interp1d(x, y, kind=kind, axis=0)
            row.append(interp)
        grid.append(row)

    # Render video.
    video_out = imageio.get_writer(
        mp4, mode="I", fps=60, codec="libx264", **video_kwargs
    )
    for frame_idx in tqdm(range(num_keyframes * w_frames)):
        imgs = []
        for yi in range(grid_h):
            for xi in range(grid_w):
                interp = grid[yi][xi]
                w = torch.from_numpy(interp(frame_idx / w_frames)).to(device)
                img = G.synthesis(ws=w.unsqueeze(0), noise_mode="const")[0]
                imgs.append(img)
        video_out.append_data(
            layout_grid(torch.stack(imgs), grid_w=grid_w, grid_h=grid_h)
        )
    video_out.close()


# ----------------------------------------------------------------------------


def parse_range(s: Union[str, List[int]]) -> List[int]:
    """Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    """
    if isinstance(s, list):
        return s
    ranges = []
    range_re = re.compile(r"^(\d+)-(\d+)$")
    for p in s.split(","):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------


def parse_tuple(s: Union[str, Tuple[int, int]]) -> Tuple[int, int]:
    """Parse a 'M,N' or 'MxN' integer tuple.

    Example:
        '4x2' returns (4,2)
        '0,1' returns (0,1)
    """
    if isinstance(s, tuple):
        return s
    m = re.match(r"^(\d+)[x,](\d+)$", s)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f"cannot parse tuple {s}")


# ----------------------------------------------------------------------------
# TODO: copied from petals_process_v1


def avg_edge_pixels_above_threshold(img, threshold=250):
    r = img[:, :, 0].astype("float")
    g = img[:, :, 1].astype("float")
    b = img[:, :, 2].astype("float")
    arr = (r + g + b) / 3
    edges = np.concatenate([arr[0, :-1], arr[:-1, -1], arr[-1, ::-1], arr[-2:0:-1, 0]])
    avg_edge = np.around(np.average(edges))
    if avg_edge < threshold:
        print(f"Average edge pixels below threshold: {avg_edge}")
        return False
    return True
