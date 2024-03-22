import os
import shutil
from glob import glob

import numpy as np

#import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
#import plotly.express as px

import torch 
from torch.utils.data import Dataset 
from torchvision.transforms import ToTensor, Resize, Compose 
from torch.utils.data import DataLoader

#import skimage as ski

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

class SSHSPH_MY(Dataset):
    """
    Desc : Dataset class for loading Remote Sensing Images (captured in Malaysia)
    The respository contain .jpg images with no labels. Hence the dataset will
    only return values for X.
    """
    def __init__(self, paths:List[str], transforms:Compose = None):
        """
        Desc : Initialization
        Parameters
            - paths : list of paths to each image
        """
        self.paths = paths
        self.transforms = transforms 

    def __len__(self):
        """
        Desc : return the size of the dataset
        """
        return len(self.paths)
    
    def __getitem__(self, idx:int):
        """
        Desc : Returns a single item from dataset. Note only a single value 
        will be returned as this dataset doesnot have labels
        Parameters :
            - idx : index
        """
        image_path = self.paths[idx]
        image = np.asarray(Image.open(image_path)) #(512,512,3)
        image_copy = np.copy(image) # Need to copy image to make it writable

        if self.transforms:
            return self.transforms(image_copy)
        else:
            return image_copy.astype("float")

def get_mean_std(dataset:str,image_paths:List[str],batch_size:int=512):
    """
    Computes mean and std for specified dataset
    Parameters
        - dataset : name of dataset (i.e "sshsph_my")
    """
    #VAR[X] = E[X**2] - E[X]**2 , where E is the expectation & VAR is variance
    c_sum,c_sq_sum, num_batches = 0,0,0

    if dataset == "sshsph_my":
        d = SSHSPH_MY(image_paths, transforms = Compose([ToTensor()])) 
        dl = DataLoader(d, batch_size = batch_size) 
        for data in dl: #Here we dont have targets, if you do its for data,_ in loader
            c_sum += torch.mean(data, dim=[0,2,3]) #Since we have (b,c,w,h)
            c_sq_sum += torch.mean(data**2, dim = [0,2,3])
            num_batches += 1

        mean = c_sum / num_batches
        std = (c_sq_sum/num_batches - mean**2)**0.5
    else:
        print("Dataset isnt availble or not implemented")
    return mean,std 

# -------------------- To get resnet50 model from RSP repo ------------------- #
def load_rsp_weights(model, path_to_weights = "pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth", num_classes = 51):
    """
    Desc : Loads pretrained weights to a resnet 50 model from the RSP repository.
    Thwe weight file (rsp-aid-resnet-50-e300-ckpt.pth) consists of a linear layer with an output of 51 hence we have to set num_classes to 51
    Inputs 
        - path_to_weights : path to the file containing weight (last layer is a Linear Layer with 51 neurons)
        - num_classes : number of classes, for the weight file (rsp-aid-resnet-50-e300-ckpt.pth) we need to set num classes to 51
    Outputs
        - res50 : Resnet50 pretrained model
    """
    res50 = model(num_classes = num_classes)
    res50_state = torch.load(path_to_weights) 
    res50.load_state_dict(res50_state["model"]) # we can add argument .load_state_dict( ... , strict = False) if the weights dont load properly, random weights will be intialised for the weights that do not work
    return res50

if __name__ == "__main__":
    
    #* -------------------------- Cleaning image dataset -------------------------- #
    #clean_image_dataset()

    #* ------------------------ using RemSensDataset class ------------------------ #
    # # There are files that consist of a single channel (grayscale) in the RGB images : Remove these
    # img_paths = glob("data/train/*")
    # img_paths.sort()
    # #remove_images(img_paths)

    # # Check Image Dataset
    # remsens_data = RemSensDataset(img_paths, resize_dims=(10000,10000))
    # train_split, test_split = torch.utils.data.random_split(remsens_data, [144,20])
    
    # #! Note Python gets killed when batchsize = 16, potential reason is RAM running out.
    # #! For now batchsize = 8 is used which works fine for - Dataloading
    # trainloader = DataLoader(train_split, batch_size=15)

    # for img, lab in trainloader:
    #     print(img.shape, lab.shape)

    #* ------------------------ Using SSHPSH Dataset class ------------------------ #
    # image_paths = glob("data/SSHSPH-RSMosaics-MY-v2.1/images/channel3_p/*")
    # sshsph_my = SSHSPH_MY(
    #     image_paths , 
    #     transforms = Compose([ToTensor(), Resize((256,256))])
    # )
    # img = sshsph_my.__getitem__(1)

    # -------------------------- test Normalizing values ------------------------- #
    dataset = "sshsph_my"
    image_paths = glob("data/SSHSPH-RSMosaics-MY-v2.1/images/channel3_p/*")
    # means, stds = get_norm_vals(dataset, image_paths)
    # print(means, stds)
    # Load SSHSPH_MY dataset

    mean, std = get_mean_std("sshsph_my", image_paths, 512)

    print(mean)
    print(std)



