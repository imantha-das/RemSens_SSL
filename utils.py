import os
import shutil
from glob import glob

import numpy as np

import cv2 as cv
import matplotlib.pyplot as plt
import plotly.express as px

import torch 
from torch.utils.data import Dataset 
from torchvision.transforms import ToTensor, Resize 
from torch.utils.data import DataLoader

from typing import List

from termcolor import colored

def clean_image_dataset():
    """
    Construct an image dataset by moving all .tif files
    - Constructs data folder to store all mosiacs data
    - Within the data folder a folder named "train" to store all .tif images and "archive" to store all other data.
    """
    # Construct data folder
    if not os.path.exists("data"):
        os.mkdir("data")
    # Construct train folder
    if not os.path.exists("data/train"):
        os.mkdir("data/train")
    # construct archive folder
    if not os.path.exists("data/archive"):
        os.mkdir("data/archive")

    for f in glob("*"):
        if f.endswith(".tif"):
            shutil.move(src = f, dst = "data/train")
        if f.endswith(".enp") or f.endswith(".ovr") or f.endswith("pdf") or f.endswith(".prj") or f.endswith(".tfw") or f.endswith("tiles") or f.endswith(".xml") or f.endswith(".zip"):
            shutil.move(src = f, dst = "data/archive")

def remove_images(paths:list):
    """
    Removes Images that are ...
        - Not of type ndarray (incomplete info, in this case grayscale)
        - Images that donot have 3 channels
    Inputs
        - list of paths to images
    """
    nd_array_cnt = 0 # 
    other_cnt = 0
    img3_cnt = 0 # 3D image count
    img3not_cnt = 0 # anything that isnt 3D image count
    for p in img_paths:
        img = cv.imread(p)
        # Check whether data is of type ndarray (as there are nontype data in the dataset)
        if isinstance(img, np.ndarray):
            nd_array_cnt += 1
            # Check if the data is 3 Dimensional
            if img.shape[2] == 3:
                img3_cnt += 1
            # Move any non 3D images to archive
            else:
                img3not_cnt += 1
                print(f"Not an 3D img : {p}")
                shutil.move(src = p , dst = "data/archive")
        # Check if the data is of type 'NoneType' 
        else:
            print(f"None type images : {p}")
            print("Moving to archive ...")
            shutil.move(src = p , dst = "data/archive")
            other_cnt += 1

    print(f"Clean image count : {nd_array_cnt}")
    print(f"Damaged images : {other_cnt}")

class RemSensDataset(Dataset):
    """
    Construct a torch dataset from remote sensing images
    - Converts Images to Torch tensors
    - Dimensions donot need to be adjusted as cv.imread outputs (C,W,H)
    - Resize images to defined size
    - Sets a label of 1 to all images for SSL
    Inputs
        - paths : A list of paths to every image in the folder
        - resize_dims : Height Width dimension resize values.
            - A value of (10000,10000) was used as large portion values were within this range
    Outputs
        - image and label
    """
    def __init__(self, paths:List[str],resize_dims:tuple[int,...] = (10000,10000)):
        self.paths = paths 
        self.resize = Resize((resize_dims))

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx:int):
        # Get Image path
        img_path = self.paths[idx]
        
        # Convert Image to Tensor
        img = cv.imread(img_path) #(H,W,C) <- ndarray
        img = ToTensor()(img) # (C,H,W) <- torch.tensor

        # Resize Image
        # Each image has a different height and width and we need each batch to have the same 
        # width and height
        img = self.resize(img)

        # As we doing self supervised learning all labels will have a value of 1, True
        label = torch.tensor([1], dtype = torch.float32)

        return img, label


if __name__ == "__main__":
    
    # Clean image dataset
    #clean_image_dataset()

    # There are files that consist of a single channel (grayscale) in the RGB images : Remove these
    img_paths = glob("data/train/*")
    img_paths.sort()
    #remove_images(img_paths)

    # Check Image Dataset
    remsens_data = RemSensDataset(img_paths, resize_dims=(10000,10000))
    train_split, test_split = torch.utils.data.random_split(remsens_data, [144,20])
    
    #! Note Python gets killed when batchsize = 16, potential reason is RAM running out.
    #! For now batchsize = 8 is used which works fine for - Dataloading
    trainloader = DataLoader(train_split, batch_size=8)

    for img, lab in trainloader:
        print(img.shape, lab.shape)




