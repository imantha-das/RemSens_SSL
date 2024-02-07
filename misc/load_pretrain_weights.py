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
sys.path.append("/home/imantha/workspace/RemSens_SSL/RSP/Scene Recognition")

import os
from models import build_model
from models.resnet import resnet50, ResNet

# Check if cuda is available
#print(colored(torch.cuda.is_available(), "green"))

res50 = resnet50()
# Print layers and weights of the model
#print(summary(res50))

#print(os.listdir("pretrained/aid28"))
pretrained_weights_path = "pretrain_weights/aid28/rsp-resnet-50-ckpt.pth"
res50.load_state_dict(torch.load(pretrained_weights_path), strict=False)

# Get SSH hostname
host_name = socket.gethostname()
  

# ==============================================================================
# Plot image 
# ==============================================================================
if host_name == "cora":
    img_dir = "/home/imantha/data/Million-AID/train/agriculture_land/arable_land/dry_field"
else:
    img_dir = "data/MillionAID_partial/train/agriculture_land/arable_land/dry_field"

img = Image.open(os.path.join(img_dir, "P0335343.jpg"))
plt.imshow(img)
#plt.show() #to display image

# ==============================================================================
# Convert image to tensor
# ==============================================================================

transform = transforms.Compose([
    transforms.PILToTensor()
])
img_t = transform(img)
img_t = img_t.unsqueeze(dim = 0).float() # Add batch dim 
print(colored(f"Tensor shape : {img_t.shape}", "green"))

# ==============================================================================
# Run Pretrained Model
# ==============================================================================

out = res50(img_t)
print(colored(f"output shape : {out.shape}", "magenta"))
print(out)