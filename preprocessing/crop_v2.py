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
import image_utils as itools

class Cropper:
  # lower_bound = np.array([180,180,180]) # really nice on orange
  satval = 5
  contval = (3, 8)
  thresh = 210
  pad = 50
  contrast_threshold = 2.5

  smallest_bbox = None
  largest_bbox = None

  def threshold(self, img):
    # blur the image to smmooth out the edges a bit, also reduces a bit of noise
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # convert the image to grayscale 
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    # apply thresholding to conver the image to binary format
    # after this operation all the pixels below 200 value will be 0
    _, gray = cv2.threshold(gray, self.thresh , 255, cv2.CHAIN_APPROX_NONE)
    return gray
  
  def process_dir(self, dir):
    itools.process_dir(dir, self.process_image)

  def get_bbox_size(self, bbox):
    x, y, width, height = bbox
    return width * height

  def find_bbox_boundaries(self,dir):
    itools.process_dir(dir, self.update_bbox_boundaries)
    print("smallest bbox", self.smallest_bbox)
    print("largest bbox", self.largest_bbox)
    
  def update_bbox_boundaries(self, file):
    img = itools.read_file(file)
    contour = itools.find_largest_threshold_contour(img, self.thresh)
    if contour is not None:
      bbox = itools.bbox_contour(contour, self.pad)
      if bbox is not None:
        if self.smallest_bbox is None or self.largest_bbox is None:
          self.smallest_bbox = bbox
          self.largest_bbox = bbox
        else:
          size = self.get_bbox_size(bbox)
          if size < self.get_bbox_size(self.smallest_bbox):
            self.smallest_bbox = bbox
          elif size > self.get_bbox_size(self.largest_bbox):
            self.largest_bbox = bbox

  def crop(self, img):
    # contrast = itools.contrast(img, self.contval)
    sat = itools.saturate(img, self.satval)
    contour = itools.find_largest_threshold_contour(sat, self.thresh)
    if contour is not None:
      bbox = itools.bbox_contour(contour, self.pad)
      if bbox is not None:
        return itools.crop(img, bbox)

  def process_image(self, file, output):
    img = itools.read_file(file)
    # contrast = itools.contrast(img, self.contval)
    sat = itools.saturate(img, self.satval)
    contour = itools.find_largest_threshold_contour(sat, self.thresh)
    if contour is not None:
      bbox = itools.bbox_contour(contour, self.pad)
      if bbox is not None:
        cropped = itools.crop(img, bbox)
        if itools.contrast(cropped) < self.contrast_threshold:
          print("IMAGE CONTRAST TOO LOW")
        else: 
          itools.write_file(cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR), output)
    else:
      print("CONTOUR OR BBOX NONT FOUND")