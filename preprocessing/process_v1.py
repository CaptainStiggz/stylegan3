# import pixellib
# from pixellib.instance import instance_segmentation

# segment_image=instance_segmentation()
# segment_image.load_model("mask_rcnn_coco.h5")
# segment_image.segmentImage("5a/ST05_SE010107.jpg", 
#   extract_segmented_objects=True,
#   save_extracted_objects=True, 
#   show_bboxes=True,
#   output_image_name="output.jpg"
# )

import cv2
import numpy as np
import os
import random
import shutil

class Processor:
  # lower_bound = np.array([180,180,180]) # really nice on orange
  lower_bound = np.array([200,200,200]) # really nice on orange
  upper_bound = np.array([255,255,255])
  pad = 50
  limit = 20

  def rand_image(self, dir):
    return self.read_file(dir+"/"+random.choice(os.listdir(dir)))

  # experiment 2
  def to_grayscale(self, img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
  
  def threshold(self, img):
    # blur the image to smmooth out the edges a bit, also reduces a bit of noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # convert the image to grayscale 
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # apply thresholding to conver the image to binary format
    # after this operation all the pixels below 200 value will be 0
    _, gray = cv2.threshold(gray, 200 , 255, cv2.CHAIN_APPROX_NONE)
    return gray

  def create_contour_mask(self, img):
    contour = self.find_largest_contour(img)
    # create a black `mask` the same size as the original grayscale image 
    mask = np.zeros_like(img)
    # fill the new mask with the shape of the largest contour
    # all the pixels inside that area will be white 
    cv2.fillPoly(mask, [contour], 255)

    # create a copy of the current mask
    res_mask = np.copy(mask)
    res_mask[mask == 0] = cv2.GC_BGD # obvious background pixels
    res_mask[mask == 255] = cv2.GC_PR_BGD # probable background pixels
    res_mask[mask == 255] = cv2.GC_FGD # obvious foreground pixels

    # create a mask for obvious and probable foreground pixels
    # all the obvious foreground pixels will be white and...
    # ... all the probable foreground pixels will be black
    mask2 = np.where(
        (res_mask == cv2.GC_FGD) | (res_mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype('uint8')

    # create `new_mask3d` from `mask2` but with 3 dimensions instead of 2
    new_mask3d = np.repeat(mask2[:, :, np.newaxis], 3, axis=2)
    mask3d = new_mask3d
    mask3d[new_mask3d > 0] = 255.0
    mask3d[mask3d > 255] = 255.0
    # apply Gaussian blurring to smoothen out the edges a bit
    # `mask3d` is the final foreground mask (not extracted foreground image)
    return cv2.GaussianBlur(mask3d, (5, 5), 0), mask2

  def apply_contour_mask(self, img, mask):
    # create the foreground image by zeroing out the pixels where `mask2`...
    # ... has black pixels
    foreground = np.copy(img).astype(float)
    foreground[mask == 0] = 255
    return foreground.astype(np.uint8)

  def find_largest_contour(self, image):
    """
    This function finds all the contours in an image and return the largest
    contour area.
    :param image: a binary image
    """
    image = image.astype(np.uint8)
    contours, hierarchy = cv2.findContours(
        image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )
    largest_contour = max(contours, key=cv2.contourArea)
    return largest_contour
  
  def apply_new_background(self, mask3d, foreground, save_name):
    """
    This function applies a new background to the extracted foreground image
    if `--new-background` flag is `True` while executing the file.
    :param mask3d: mask3d mask containing the foreground binary pixels
    :param foreground: mask containg the extracted foreground image
    :param save_name: name of the input image file
    """
    # normalization of mask3d mask, keeping values between 0 and 1
    mask3d = mask3d / 255.0
    # get the scaled product by multiplying
    foreground = cv2.multiply(mask3d, foreground)
    # read the new background image
    background = cv2.imread('input/background.jpg')
    # resize it according to the foreground image
    background = cv2.resize(background, (foreground.shape[1], foreground.shape[0]))
    background = background.astype(np.float)
    # get the scaled product by multiplying
    background = cv2.multiply(1.0 - mask3d, background)
    # add the foreground and new background image
    new_image = cv2.add(foreground, background)
    return new_image
  
  # experiment 1

  def process_dir(self, dir):
    out = dir+"_out"
    if os.path.exists(out):
      shutil.rmtree(out)
    os.makedirs(out)

    i = 0
    for file in os.listdir(dir):
      if file.endswith(".jpg"):
        self.process_image(dir+"/"+file, out+"/"+file)
        i += 1
        if self.limit > 0 and i > self.limit:
          break

  def process_image(self, file, output):
    img = self.read_file(file)
    mask = self.mask(img)
    crop = self.bbox_crop_mask(mask)
    if crop is not None:
      img = self.apply_mask(img, mask)
      img = self.crop(img, crop)
      img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
      self.write_file(img, output)
    else:
      print("skipping", file)
  
  def read_file(self, file):
    img = cv2.imread(file)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  def write_file(self, img, file):
    return cv2.imwrite(file, img)

  def mask(self, img):
    return cv2.inRange(img, self.lower_bound, self.upper_bound)

  def apply_mask(self, img, mask):
    masked_img = np.copy(img)
    masked_img[mask != 0] = [255, 255, 255]
    return masked_img
  
  def crop(self, img, crop):
    x, y, w, h = crop
    return img[y:y+h,x:x+w]

  def bbox_crop_mask(self, img):
    """Crop the mask"""
    contours,_ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for cnt in contours:
        _x,_y,_w,_h = cv2.boundingRect(cnt)
        if _w > 100 and _h > 100:
            x,y,w,h = _x, _y, _w, _h
            diff = w - h
            if diff > 0:
                x -= self.pad
                w += 2 * self.pad
                y = y - diff // 2 - self.pad
                h = w
            elif diff < 0:
                y -= self.pad
                h += 2 * self.pad
                x = x + diff // 2 - self.pad
                w = h
            return x,y,w,h
  
  def saturate(self, img, satval):
    pass

# processor = Processor()
# processor.process_image("5a/ST05_SE010122.jpg")