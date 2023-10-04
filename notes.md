# General Info

- 24 images left in archives/tiles that havent been used
  - Can we use them or not ?
  - Are they just larger sections of the images
- 164 training images avaialble, ~30GB

# Data Cleaning

- Files removed due to single channel (Grayscale)
  - data/train/20150727_02_dsm.tif
  - data/train/20160815_04_dsm.tif

# Issues

- Unable to run RemSensDataset class if batchsize is greater than or equal to 16
- Potential issue is that there isnt enough memory (RAM)
