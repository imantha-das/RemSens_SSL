# ==============================================================================
# Desc : utility functions for BigEarthNet dataset
# ==============================================================================
import os
import json
from typing import List, Tuple
from termcolor import colored

from PIL import Image
import pickle

import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer


def get_encodings(image_labels:List[str] | List[List[str]],unique_labels:List[str],label_encodings:bool = True):
    """
    Desc : Get label encordings for each image in the form of torch tensors
    Inputs
        - image_labels : label for each image
        - unique_labels : all unique labels available in BigEarthNet (There is 43 ...)
        - label_encordings : if true returns label encordings else multilabel format
            - when the encorder is a LabelEncoderpass a list for image_labels
            - when the encoder is a MultiLabelBinarizer pass list of lists
    Outputs
        - encordings : encordings for labels
        - encorder : scikit learn trained enorder
    """
    if label_encodings: 
        encoder = LabelEncoder()
        encoder.fit(unique_labels)
        encodings = encoder.transform(image_labels)
    else:
        encoder = MultiLabelBinarizer(classes = unique_labels)
        encodings = encoder.fit_transform(image_labels)

    return encodings, encoder

class BigEarthNet(Dataset):
    """Dataset class to load Multi Label dataset BigEarthNet"""
    def __init__(self, paths:List[str], unique_labels:list, encodings : str)->Tuple[torch.tensor,List[int], List[str]]:
        """
        Inputs 
            - paths : a list of paths to every image in respective folder
            #* This will be likely the "train" folder for example
            - unique_labels : A list of labels of the entire dataset. There are 43 labels in BigEarthNet
            - encodings : option between "ordinal" or "ohe
                - when the encorder is a LabelEncoder pass a list for image_labels
                - when the encoder is a MultiLabelBinarizer pass list of lists
        Outputs
            - rgbt : tensor of shape (3,120,120) containg RGB values
            - encode_label : list of labels either encoded using LabelEncoder or MultiLabelBinarizer
            - 
        """
        self.paths = paths
        self.unique_labels = unique_labels
        self.encodings = encodings

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        # Path to main folder containing all tif files
        image_path = self.paths[idx]

        # Search for bands B04,B03, B02 which contain PIL images to red,green,blue channels
        red_band_path = os.path.join(image_path, [f for f in os.listdir(image_path) if f.endswith("B04.tif")][0])
        green_band_path = os.path.join(image_path, [f for f in os.listdir(image_path) if f.endswith("B03.tif")][0])
        blue_band_path = os.path.join(image_path, [f for f in os.listdir(image_path) if f.endswith("B02.tif")][0])
        
        # Read PIL Image
        #print(colored(red_band_path, "blue"))
        red_band = Image.open(red_band_path)
        green_band = Image.open(green_band_path)
        blue_band = Image.open(blue_band_path)

        # Convert Images to tensors
        rt = ToTensor()(red_band).float() #shape : (1,120,120)
        gt = ToTensor()(green_band).float() #shape (1,120,120)
        bt = ToTensor()(blue_band).float()#shape (1,120,120)
        
        # Concatenate tensors to make RGB image tensor
        rgbt = torch.cat([rt, gt, bt], dim = 0) #(3,120,120)

        # get_labels for the image
        json_file_path = os.path.join(image_path,[f for f in os.listdir(image_path) if f.endswith(".json")][0])
        with open(json_file_path) as f:
            metadata = json.load(f)

        # Get labels from metadata, this is MultiLabel Classification dataset hence the label(s)
        # But these are the labels associated with a single image
        raw_labels = metadata["labels"]
        
        if self.encodings == "ordinal":
            encodings, encoder = get_encodings(raw_labels, self.unique_labels, label_encodings = True)
        else:
            encodings, encoder = get_encodings([raw_labels], self.unique_labels, label_encodings = False)

        misc = {
            "raw_labels" : raw_labels,
            "encoder" : encoder
        }
        
        #print(colored(rgbt.shape, "blue"), colored(encodings.shape, "magenta"))
        if self.encodings == "ohe":
            assert (rgbt.shape == (3,120,120)) or (encodings.shape == (1,43))

        

        return rgbt, encodings

if __name__ == "__main__":
    
    with open("bigearthnet_utils/bigearthnet_unique_labels.json") as f:
        unique_labels = json.load(f)["labels"]

    # Test get_encodings
    img1_labs = ["Transitional woodland/shrub"]
    img2_labs = ["Salt marshes", "Sclerophyllous vegetation", "Transitional woodland/shrub"]

    # encodings, encorder = get_encodings([img2_labs], unique_labels, label_encodings = False)
    # print(encodings)

    # label_encoder = LabelEncoder()
    # label_encoder.fit(unique_labels)
    # pickle.dump(label_encoder, open("bigearthnet_utils/label_encoder.pkl", "wb"))
    # print(label_encoder.transform(["Transitional woodland/shrub"]))

    # label_encoder = pickle.load(open("bigearthnet_utils/label_encoder.pkl", "rb"))
    # print(label_encoder.transform(img2_labs))

    # mlb_encoder = MultiLabelBinarizer()
    # mlb_encoder.fit(unique_labels)
    # pickle.dump(mlb_encoder, open("bigearthnet_utils/mlb_encoder.pkl", "wb"))
    # print("intial encodeing ...")
    # print(mlb_encoder.transform([img1_labs]))

    # new_mlb_encoder = pickle.load(open("bigearthnet_utils/mlb_encoder.pkl", "rb"))
    # print("loaded encodeing ...")
    # print(new_mlb_encoder.transform([img2_labs]))

    # parent_dir = "data/BigEarthNet-v1.0/train"
    # image_paths = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)]
    
    # big_earth_net = BigEarthNet(image_paths, unique_labels, encodings = "ohe")
    # img2, label, misc = big_earth_net.__getitem__(2)
    # print(img2.shape, label)

    # # Check if encoded labs are infact true to the raw labels
    # print(f"raw_labels : {misc['raw_labels']}")
    # print(f"invers transform :{misc['encoder'].inverse_transform(label)}")

