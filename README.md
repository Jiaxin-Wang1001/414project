For this project, we have taken the code from the repo at <https://github.com/hzxie/Pix2Vox> as our starting point. We modified the config and the decoder, disabled the merger and refiner, turning it into a general 3D reconstruction network before we tested our new loss functions on it. We also modified the utilities to suit our 2D projection losses

The original ShapeNet data set does not contain orthographic projection of the groundtruth. We wrote a helper function in projection_genenator.py to generate them. Change the paths then run projection_genenator.py to generate the necessary data.

Change the data paths in config.py to your paths
__C.DATASETS.SHAPENET.TAXONOMY_FILE_PATH    = '/content/drive/MyDrive/414project/datasets/ShapeNet.json'
__C.DATASETS.SHAPENET.RENDERING_PATH        = '/content/drive/MyDrive/ShapeNetRendering/%s/%s/rendering/%02d.png'
__C.DATASETS.SHAPENET.VOXEL_PATH            = '/content/drive/Shareddrives/CMPUT_414/ShapeNetVox32/%s/%s/model.binvox'
__C.DATASETS.SHAPENET.PROJECTION_PATH       = '/content/drive/Shareddrives/CMPUT_414/ShapeNetVox32/%s/%s/%01d.png'


To use the two headed predictor, change line 69 in train.py to use 'decoder = Decoder2(cfg)'
Then change line 169 to 'encoder_loss = encoder_loss1 + encoder_loss2'


To use all other predictors, change it to 'decoder = Decoder(cfg)'
  To use simple max pooling projection loss change line 169 to 'encoder_loss = encoder_loss2'
  To use the summation loss, change it to 'encoder_loss = encoder_loss1 + encoder_loss2'
  To use the alternating loss, comment out line 169 and uncomment lines 170-173
  To use the baseline setting, change line 169 to 'encoder_loss = encoder_loss1'

Then to train the models, run python3 runner.py. The models will be save at ./output