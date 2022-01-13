import os
import json

def find_best_checkpoint(dir):
  best = None
  checkpoint = None
  for root, dirs, files in os.walk(dir, topdown=False):
    for name in files:
      if name.endswith("metric-fid50k_full.jsonl"):
        file_path = f'{root}/{name}'
        with open(file_path, 'r', encoding='utf-8') as f:
          for line in f:
            line = json.loads(line)
            fid = line['results']['fid50k_full']
            if best is None or fid < best:
              best = fid
              cp = line['snapshot_pkl']
              checkpoint = f'{root}/{cp}'
  print(fid, checkpoint)
  return (fid, checkpoint)

        
        