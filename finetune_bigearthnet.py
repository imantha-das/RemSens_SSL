import os
import json
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader
from bigearthnet_utils.ben_utils import BigEarthNet 


if __name__ == "__main__":
    
    # Load json file containing all labels of Bigearthnet
    with open("bigearthnet_utils/bigearthnet_unique_labels.json") as f:
        unique_labels = json.load(f)["labels"]

    # Parent directory containing all training images and label
    train_dir = "data/BigEarthNet-v1.0/train"
    valid_dir = "data/BigEarthNet-v1.0/valid"
    train_image_paths = [os.path.join(train_dir, f) for f in os.listdir(train_dir)]
    valid_image_paths = [os.path.join(valid_dir, f) for f in os.listdir(valid_dir)]

    # Construct Dataset and DataLoader
    train_dataset = BigEarthNet(train_image_paths, unique_labels, encodings = "ohe")
    valid_dataset = BigEarthNet(valid_image_paths, unique_labels, encodings = "ohe")

    #print(f"Train dataset size : {len(train_dataset)} , Validation dataset size : {len(valid_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size = 64, drop_last = True)
    valid_loader = DataLoader(valid_dataset, batch_size = 64)

    img2, lab2 = train_dataset.__getitem__(2)
    print(img2.shape, lab2.shape)



