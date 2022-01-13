import image_utils as itools
import os
import shutil
import re
import cv2 
import json
import numpy as np
from PIL import Image

class PetalProcessor:
  dir = ""
  deleted_dir = dir + "Deleted"
  duplicate_dir = f'{deleted_dir}/duplicates'
  empty_dir = f'{deleted_dir}/empty'
  bboxes = {}
  
  satval = 5
  contval = (3, 8)
  thresh = 210
  pad = 50
  contrast_threshold = 2.5

  smallest_bbox = None
  largest_bbox = None
  smallest_petal = None
  largest_petal = None

  def __init__(self, dir):
    self.dir = dir
    self.deleted_dir = dir + "Deleted"
    self.duplicate_dir = f'{self.deleted_dir}/duplicates'
    self.empty_dir = f'{self.deleted_dir}/problems'
    self.out_dir = dir + "Out"

  def deduplicate(self):
    """Deduplicate images"""
    itools.run_on_dir(self.dir, self.deduplicate_image)


  def delete_img(self, file, path):
    filepath = file[len(self.dir):]
    dest = f'{path}/{filepath}'
    if not os.path.exists(os.path.dirname(dest)):
      os.makedirs(os.path.dirname(dest))
    shutil.move(file, dest)
  
  def save_output_img(self, img, file):
    filepath = file[len(self.dir):]
    dest = f'{self.out_dir}/{filepath}'
    if not os.path.exists(os.path.dirname(dest)):
      os.makedirs(os.path.dirname(dest))
    itools.write_file(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), dest)

  def save_json_file(self):
    with open(self.dir + "boxes.json", 'w') as outfile:
      json.dump(self.bboxes, outfile)

  def deduplicate_image(self, file):
    if self.deleted_dir in file:
      return
    x = re.search('-([-+]?[0-9]+).jpg$', file)
    if x != None:
      print(f'Found duplicate: {file}. Deleting...')
      self.delete_img(file, self.duplicate_dir)

  def resize_all(self):
    itools.run_on_dir(self.dir, self.resize_img)

  def resize_img(self, file, size = 1024):
    if self.deleted_dir in file or self.out_dir in file:
      return
    filepath = file[len(self.dir):]
    if os.path.exists(f'{self.out_dir}/{filepath}'):
      return

    with Image.open(file) as img:
      resized = img.resize((1024, 1024), resample=Image.BICUBIC)
      filepath = file[len(self.dir):]
      dest = f'{self.out_dir}/{filepath}'
      if not os.path.exists(os.path.dirname(dest)):
        os.makedirs(os.path.dirname(dest))
      # itools.write_file(cv2.cvtColor(img, cv2.COLOR_RGB2BGR), dest)
      resized.save(dest)      

  def crop_all_to_largest_square(self):
    json_path = f'{self.dir}boxes.json'
    if os.path.exists(json_path):
      f = open(f'{self.dir}boxes.json')
      self.bboxes = json.load(f)
    else:
      self.bboxes = {}
    itools.run_on_dir(self.dir, self.crop_to_largest_bbox)

  def get_bbox_size(self, bbox):
    x, y, width, height = bbox
    return width * height
    
  def delete_empty(self, file):
    print(f'Found empty: {file}. Deleting...')
    self.delete_img(file, self.empty_dir)

  def find_largest_bbox(self, img, bbox):
    img_height, img_width, _channels = img.shape
    x, y, w, h = bbox
    diff = w - h
    if diff > 0:
      y = y - diff // 2
      h = w
    elif diff < 0:
      x = x + diff // 2
      w = h
    if w != h:
      print("BBOX IS NOT SQUARE!")
      return None
    
    dx = x
    dy = y
    dw = img_width - (x + w)
    dh = img_height - (y + h)
    pad = 40
    if dx <= pad or dy <= pad or dh <= pad or dw <= pad:
      print("BBOX TOO CLOSE TO EDGE")
      return None

    offset = min(dx, dy, dw, dh)
    return (x - offset, y - offset, w + 2 * offset, h + 2 * offset)

  def crop_to_largest_bbox(self, file):
    if self.deleted_dir in file or self.out_dir in file:
      return
    filepath = file[len(self.dir):]
    if os.path.exists(f'{self.out_dir}/{filepath}'):
      return
      
    img = itools.read_file(file)
    contrast = itools.contrast(img, self.contval)
    sat = itools.saturate(contrast, self.satval)
    gray = itools.threshold(sat, self.thresh)
    # return (gray, None, None)
    try:
      contour = itools.find_largest_contour(gray)
    except:
      print("CONTOUR ERRORED")
      self.delete_empty(file)
      return


    if contour is None:
      self.delete_empty(file)
      return

    # get the bounding box
    bbox = cv2.boundingRect(contour)

    # delete if there is no bounding box
    if bbox is None:
      print("NO BBOX FOUND")
      self.delete_empty(file)
      return

    if self.get_bbox_size(bbox) < 250000:
      print("BBOX TOO SMALL")
      self.delete_empty(file)
      return

    crop = self.find_largest_bbox(img, bbox)
    if crop is None:
      self.delete_empty(file)
      return

    cropped = itools.crop(img, crop)
    
    self.bboxes[filepath] = bbox
    self.save_json_file()
    self.save_output_img(cropped, file)

  def process_image_bbox(self, file, func):
    img = itools.read_file(file)
    contour = itools.find_largest_threshold_contour(img, self.thresh)
    if contour is None:
      self.delete_empty(file)
      return

    # get the bounding box
    bbox = itools.bbox_contour(contour, self.pad)

    # delete if there is no bounding box
    if bbox is None:
      self.delete_empty(file)
      return
    
    func(file, img, bbox)

  def find_dark_files(self):
    itools.run_on_dir(self.dir, self.avg_img_edge_pixels)

  def avg_img_edge_pixels(self, file):
    img = itools.read_file(file)
    r = img[:,:,0].astype('float')
    g = img[:,:,1].astype('float')
    b = img[:,:,2].astype('float')
    arr = (r + g + b) / 3
    edges = np.concatenate([arr[0,:-1], arr[:-1,-1], arr[-1,::-1], arr[-2:0:-1,0]])
    avg_edge = np.around(np.average(edges))
    if avg_edge < 250:
      print(avg_edge, file)
      self.delete_empty(file)