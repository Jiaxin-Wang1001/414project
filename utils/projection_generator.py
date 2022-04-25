import os
import json
import numpy as np
import imageio
from utils import binvox_rw
PROJECTION_PATH       = '/content/drive/Shareddrives/CMPUT_414/ShapeNetVox32/%s/%s/'
VOXEL_PATH            = '/content/drive/Shareddrives/CMPUT_414/ShapeNetVox32/%s/%s/model.binvox'

def generate(out_path, in_path):
  
  with open(in_path, 'rb') as f:
      m1 = binvox_rw.read_as_coord_array(f)
  data = m1.data
  temp1 = np.ndarray.copy(data)
  temp2 = np.ndarray.copy(data)
  temp3 = np.ndarray.copy(data)
  temp1[0,:] = 0
  temp2[1,:] = 0
  temp3[2,:] = 0

  x = binvox_rw.sparse_to_dense(temp1, m1.dims)[0,:,:].astype(np.uint8)
  y = binvox_rw.sparse_to_dense(temp2, m1.dims)[:,0,:].astype(np.uint8)
  z = binvox_rw.sparse_to_dense(temp3, m1.dims)[:,:,0].astype(np.uint8)
  x[x == 1] = 225
  y[y == 1] = 225
  z[z == 1] = 225

  imageio.imwrite(out_path + '1.png', x)
  imageio.imwrite(out_path + '2.png', y)
  imageio.imwrite(out_path + '3.png', z)



with open("/content/drive/Shareddrives/CMPUT_414/Pix2Vox/datasets/ShapeNetLite.json", encoding='utf-8') as f:
  data = json.loads(f.read())
data = data[1:]
for t in data:
  temp = t["train"] + t["test"] + t["val"]
  folder_name = t['taxonomy_id']
  print("projecting taxonomy", folder_name)
  i = 1
  for sample_idx, sample_name in enumerate(temp):
    print("projecting item", sample_name, i)
    # Get file path of volumes
    volume_file_path = VOXEL_PATH % (folder_name, sample_name)
    projection_file_path = PROJECTION_PATH % (folder_name, sample_name)
    if not os.path.exists(projection_file_path + '1.png'):
      generate(projection_file_path, volume_file_path)
    i+=1