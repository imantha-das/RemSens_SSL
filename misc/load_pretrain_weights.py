# ==============================================================================
# Desc : Loads pretrained weights from RSP Github Repository
# Author : Imantha 
# Date : 07/02/2024 
# ==============================================================================

import sys 
import os
from termcolor import colored

import torch
from torchsummary import summary
from torchvision import transforms

import socket
from PIL import Image
import matplotlib.pyplot as plt

# Path to RSP Scene Recognition - Pretrained Models saved here
sys.path.append("RSP/Scene Recognition")

import os
from models import build_model
from models.resnet import resnet50, ResNet

# Check if cuda is available
#print(colored(torch.cuda.is_available(), "green"))

res50 = resnet50()
# Print layers and weights of the model
#print(summary(res50))

# Load Model weights
pretrained_weights = "pretrain_weights/aid28/rsp-resnet-50-ckpt.pth"
res50.load_state_dict(torch.load(pretrained_weights), strict=False)
  
# ==============================================================================
# Plot image 
# ==============================================================================
img_dir = "data/Million-AID/train/agriculture_land/arable_land/dry_field"
img_name = "P0335343.jpg"
img = Image.open(os.path.join(img_dir, img_name))
plt.imshow(img)
#plt.show() #to display image, doesnot work when SSH to cora, something wrong with image display

# ==============================================================================
# Convert image to tensor
# ==============================================================================

transform = transforms.Compose([
    transforms.PILToTensor()
])
img_t = transform(img)
img_t = img_t.unsqueeze(dim = 0).float() # Add batch dim 
#print(colored(f"Tensor shape : {img_t.shape}", "green"))

# ==============================================================================
# Run Pretrained Model
# ==============================================================================

out = res50(img_t)
print(colored(f"output shape : {out.shape}", "magenta"))
print(colored(f"Model Weights : {out}", "green"))

labs = {
    #Agriculture land
    0 : "dry_field", 1 : "greenhouse", 2 : "paddy field", 3 : "terraced field",4 : "meadow", 
    5 : "forrest", 6 : "orchard",
    # Commercial land
    7: "commercial area",
    # Public Service land
    8 : "basketball court", 9 : "tennis court", 10 : "baseball field", 11 : "ground track field", 
    12 : "golf course", 13 : "stadium", 14 :"cemetery", 15 : "church", 16 : "swimming pool",
    #Industrial land
    17 : "wastewater plant", 18 : "storage tank", 19 : "oil field", 20 : "works", 21 : "solar", 
    22 : "wind turbine", 23 : "substation", 24 : "mine", 25 : "quarry",
    # Transportation land
    26 : "apron", 27 : "helipad", 28 : "runway", 29 : "roundabout", 30 : "parking lot", 
    31 : "intersection", 32 : "bridges", 33 : "viaduct", 34 : "road", 35 : "train station", 
    36 : "railway", 37 : "pier",
    # Unutilized land
    38 : "rock land", 39 : "bare land", 40 : "ice land", 41 : "island", 42 : "desert", 
    43 : "sparse shrub",
    # Residential land
    44 : "detached house", 45 : "apartment", 46 : "mobile home park",
    # Water area
    47 : "beach", 48 : "lake", 49 : "river", 50 : "dam"
}

#print(out)
#print(torch.argmax(out), labs[torch.argmax(out.item())])

#print(labs[torch.argmax(out).item()])
