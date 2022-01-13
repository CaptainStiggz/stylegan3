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

import rawpy
import imageio
import os
import shutil
import pathlib

class Converter:
  processed = {}
  outdir = '../dataset/originals'
  indir = '/Volumes/My Passport for Mac/Original RAW Files'
  count = 0

  def get_target(self, file):
    path = file[len(self.indir):]
    outdir = path.split('/')[1]
    parts = outdir.split('_')
    stem = pathlib.Path(file).stem
    dir = self.outdir+'/'+str(int(parts[0]))+parts[1]
    return dir+'/'+stem+'.jpg'

  def file_is(self, file, suffix):
    ext = pathlib.Path(file).suffix
    return ext.lower() == suffix.lower()

  def convert(self, file):
    if self.file_is(file, ".nef"):
      target = self.get_target(file)
      if target not in self.processed:
        if not os.path.exists(os.path.dirname(target)):
          os.makedirs(os.path.dirname(target))
        with rawpy.imread(file) as raw:
          # rgb = raw.postprocess(use_camera_wb=True)
          rgb = raw.postprocess(
            # use_auto_wb=True,
            # use_camera_wb=True,
            no_auto_bright=True,
            # demosaic_algorithm=rawpy.DemosaicAlgorithm(10)
            # auto_bright_thr=0.2
            # gamma=(2.222, 4.5)
            user_wb=[1, 1, 1 ,1]
            # linear
            # gamma=(1,1), no_auto_bright=True
          )
          imageio.imsave(target, rgb)
        self.processed[target] = True
        finished = len(self.processed) - 1
        print(f'{finished} / {self.count} completed.')
        return rgb

  def convert_dir(self, dir):
    self.processed = {}
    self.indir = dir
    if not os.path.exists(self.outdir):
      os.makedirs(self.outdir)

    print("Converting files from " +dir+ " ("+self.outdir+")")
    self.load_converted(self.outdir)
    finished = len(self.processed) - 1
    print(f'{finished} files already converted.')

    self.count = 0
    for root, dirs, files in os.walk(dir, topdown=False):
      for name in files:
        if self.file_is(os.path.join(root, name), ".nef"):
          self.count += 1

    print(str(self.count - finished) + " files to convert.")
    
    for root, dirs, files in os.walk(dir, topdown=False):
      for name in files:
        self.convert(os.path.join(root, name))

  def load_converted(self, dir):
    for root, dirs, files in os.walk(dir, topdown=False):
      for name in files:
        self.processed[os.path.join(root, name)] = True