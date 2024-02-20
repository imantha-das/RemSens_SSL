# ==============================================================================
# Desc : utility functions for BigEarthNet dataset
# ==============================================================================
import os
import json
from typing import List, Tuple
import ast
from termcolor import colored

import pandas as pd

from PIL import Image
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Compose, Normalize

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer

torch.manual_seed(0)

class BigEarthNet(Dataset):
    def __init__(self, paths_and_labs:pd.DataFrame, transform = None)->Tuple[torch.Tensor, torch.Tensor]:
        """
        Class to construct a dataset from BigEarthNet
        Inputs
            paths_and_labs : Pandas Dataframe containing paths to each of the RGB channels and encoded labels
        Outputs
            rgb_t : Torch tensor containing RGB value for an image
            enc : corresponding encodings for image
        """
        self.df = paths_and_labs
        self.transform = transform

        # Check if all columns are inside dataframe
        assert all(col_name in paths_and_labs.columns.values for col_name in["red_band_path", "green_band_path", "blue_band_path", "enc_labels"]), "column not inside dataframe"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Pandas Series with path information to each band and label encordings
        image_data = self.df.iloc[idx]
        #print(f"loading {image_data['data_folder_path']} at index {idx}")

        # Read PIL Image
        r = Image.open(image_data["red_band_path"])
        g = Image.open(image_data["green_band_path"])
        b = Image.open(image_data["blue_band_path"])

        # Convert PIL to Torch Tensors
        r_t = ToTensor()(r).float()
        g_t = ToTensor()(b).float()
        b_t = ToTensor()(b).float()

        # Concatenate tensor to make one RGB tensor
        rgb_t = torch.cat([r_t, g_t, b_t], dim = 0) #(3,120,120)

        # Get encodings from dataframe
        enc = image_data["enc_labels"]
        enc = torch.Tensor(ast.literal_eval(enc))

        if self.transform:
            return self.transform(rgb_t), enc
        else:
            return rgb_t, enc


if __name__ == "__main__":
    

    # Get dataframe containing paths to RGB Images and encoded labels
    df_paths_labs = pd.read_csv("bigearthnet_utils/ben_datapaths_and_labels_v4.csv")
    df_paths_labs.drop("Unnamed: 0", axis = 1, errors = "ignore", inplace = True)

    # Load BigEarthNet Dataset    
    ben = BigEarthNet(df_paths_labs)

    # Compute train, valid and test size
    train_size = int(0.2 * len(ben))
    test_size = len(ben) - train_size
    print(f"Train size : {train_size}, Test size : {test_size}")
    ben_train, ben_test = torch.utils.data.random_split(ben, [train_size, test_size])

    # To get a Image and its label you can use .__getitem__(idx) method
    train_img, train_lab = ben_train.__getitem__(1)
    test_img, test_lab = ben_test.__getitem__(1)

    # Find the mean and std of each of the color channels for the entire train set
    loader = DataLoader(ben_train, batch_size = len(ben_train))
    data = next(iter(loader))
    
    mean = data[0].mean(dim = (0,2,3))
    std = data[0].std(dim = (0,2,3))
    print(f"Mean : {mean}, Std : {std}")

    # Now that the mean and std are available load Dataset and Normalize it.
    ben_norm = BigEarthNet(df_paths_labs, transform = Compose([Normalize(mean = mean, std = std)]))
    ben_train_norm, ben_test_norm = torch.utils.data.random_split(ben_norm,[train_size, test_size])

    # Test some images from training set to see if they have normalized
    train_img_norm, train_lab_norm = ben_train_norm.__getitem__(1)
    print(f"Max : {train_img_norm.max()}, Min :  {train_img_norm.min()}")

    train_img_norm, train_lab_norm = ben_train_norm.__getitem__(2)
    print(f"Max : {train_img_norm.max()}, Min :  {train_img_norm.min()}")

    train_img_norm, train_lab_norm = ben_train_norm.__getitem__(3)
    print(f"Max : {train_img_norm.max()}, Min :  {train_img_norm.min()}")

    # Normalization doesnt get values between -1 and 1, but are reduced to max value 4ish
    # Check with the unnormalized dataset to see any changes
    train_img, train_lab = ben_train.__getitem__(1)
    print(f"Max : {train_img.max()}, Min :  {train_img.min()}")

    train_img, train_lab = ben_train.__getitem__(2)
    print(f"Max : {train_img.max()}, Min :  {train_img.min()}")

    train_img, train_lab = ben_train.__getitem__(3)
    print(f"Max : {train_img.max()}, Min :  {train_img.min()}")
