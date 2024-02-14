import os 
import numpy as np
# ==============================================================================
# Million AID
# ==============================================================================
parent_aid_dir = "data/Million-AID/train"

cpt = sum([len(files) for r, d, files in os.walk(parent_aid_dir)])
#print(f"Total Number of files : {cpt}")

#print(list(os.walk(parent_aid_dir)))

# ==============================================================================
# BigEarthNet
# - Contains 590326 folders (images)
# - Each folder contains 12 bands (B01,B02,B03,B04,B05,B06,B07,B08,B08A,B09,B11,B12)
# - Bands B02 (red), B03 (Green), B04(Blue) are the only ones that are important for our problem
# - Each Image is 120 x 120 in size ( A sentinel tile is 1200 x 1200 which are broken into pathces for ML)
# - Also contains a metadata Json file
# - Note BigEarthNet is a Multilabel classification dataset unlike MillionAID

# Images come from 10 European countries (from July 2017 - May 2018)
# BigEarth Net doesnt include band B10
# ==============================================================================
from PIL import Image
import json
import torch
from torchvision.transforms import ToTensor, Normalize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import itertools

#Loading Dataset
# ------------------------------------------------------------------------------
# Load a random folder containing image bands and labels
path_to_train = "data/BigEarthNet-v1.0/train"
random_folder = np.random.choice(os.listdir(path_to_train))
path_to_folder = os.path.join(path_to_train, random_folder)
content = os.listdir(path_to_folder)

# Print files inside folder
for f in content:
    print(f)

# Constructing RGB image by combining bands
# ------------------------------------------------------------------------------
# Get paths to tiff files
red_band_path = os.path.join(path_to_folder, [f for f in content if f.endswith("B04.tif")][0]) #select band B04 refering red band
green_band_path = os.path.join(path_to_folder, [f for f in content if f.endswith("B03.tif")][0]) # select band B03 refering to green band
blue_band_path = os.path.join(path_to_folder, [f for f in content if f.endswith("B02.tif")][0]) # select band B02 refering to blue band

# Read PIL Image
red_band = Image.open(red_band_path)
green_band = Image.open(green_band_path)
blue_band = Image.open(blue_band_path)

# Convert Images to tensors
rt = ToTensor()(red_band).float() #shape : (1,120,120)
gt = ToTensor()(green_band).float() #shape (1,120,120)
bt = ToTensor()(blue_band).float()#shape (1,120,120)

# Find Minimum and Maximum pixels in each band
print(f"Red band value range : {rt.min()} - {rt.max()}")
print(f"Green band value range : {gt.min()} - {gt.max()}")
print(f"Blue band value range : {bt.min()} - {bt.max()}")

# COncatenate tensors to make RGB image tensor
rgbt = torch.cat([rt, gt, bt], dim = 0) #(3,120,120)

# Visualize Image
plt.imshow(rgbt.permute(1,2,0))
#plt.show()

#! NOTICE IMAGE IS EMPTY, THIS LIKELY SINCE VALUES ARE TOO HIGH

# Normalize image
# Find mean and std of each color channel to normalize
normalizer = MinMaxScaler()
rtn = normalizer.fit_transform(rt.squeeze(0).numpy()) 
gtn = normalizer.fit_transform(gt.squeeze(0).numpy()) 
btn = normalizer.fit_transform(bt.squeeze(0).numpy()) 
#rtn = (rt - rt.mean())/rt.std()

print(f"Normalized Red band value range : {rtn.min()} - {rtn.max()}")
print(f"Normalized Green band value range : {gtn.min()} - {gtn.max()}")
print(f"Normalized Blue band value range : {btn.min()} - {btn.max()}")

# Add dimension outside for concatenation
rtn = rtn[np.newaxis, :, :]
gtn = gtn[np.newaxis, :, :]
btn = btn[np.newaxis, :, :]

rgbtn = np.concatenate([rtn,gtn,btn], axis = 0)
rgbtn = np.moveaxis(rgbtn,[2,1,0],[0,1,2]) #move the 2nd axis to 0th position ...
print(rgbtn.shape)

# Getting Labels of Image
# ------------------------------------------------------------------------------
# Get labels for image
json_file = [f for f in content if f.endswith(".json")][0]
path_to_json = os.path.join(path_to_folder, json_file)
with open(path_to_json) as f:
    metadata = json.load(f)

# print("\n")
# for k,v in metadata.items():
#     print(f"{k} : {v} \n")

labels = metadata["labels"]

# plot image
plt.imshow(rgbtn)
plt.title(str(labels))
#plt.show()

# Getting a count map of labels training set
# ------------------------------------------------------------------------------
from termcolor import colored
print(colored("\nCalcualting countmap for labels in training set ...", "green"))

folders = os.listdir(path_to_train)

all_labels = []
for folder in folders:
    path_to_folder = os.path.join(path_to_train, folder)
    json_file = [f for f in os.listdir(path_to_folder) if f.endswith(".json")][0]
    path_to_json = os.path.join(path_to_folder, json_file)
    with open(path_to_json) as f:
        metadata = json.load(f)
    
    labels = metadata["labels"]
    all_labels.append(labels)

all_labels = list(itertools.chain(*all_labels))
unique_labels = np.unique(all_labels)

print(colored(f"\nThere are {len(unique_labels)} unique labels in dataset", "magenta"))


unique_labs_dict = {"labels" : list(unique_labels)}
with open("bigearthnet_utils/bigearthnet_unique_labels.json", "w") as f:
    json.dump(unique_labs_dict, f)

# ------------------------------------------------------------------------------
# from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# label_encoder = LabelEncoder()
# label_encoder.fit(unique_labels)
# print(label_encoder.classes_)

# encoded_labs = label_encoder.transform(['Broad-leaved forest', "Sea and ocean"])
# print(encoded_labs)

