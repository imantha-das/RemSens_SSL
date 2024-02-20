import os
import sys
import collections
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from bigearthnet_utils.ben_utils import BigEarthNet 

# Load path to RSP Scene Recognition repository in order to load pretrained model
sys.path.append("/home/imantha/workspace/RemSens_SSL/RSP/Scene Recognition")

#from models import build_model
from models.resnet import resnet50

def load_pretrained_weights(path_to_weights = "pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth", num_classes = 51):
    """
    Desc : Loads pretrained weights to a resnet 50 model from the RSP repository.
    Thwe weight file (rsp-aid-resnet-50-e300-ckpt.pth) consists of a linear layer with an output of 51 hence we have to set num_classes to 51
    Inputs 
        - path_to_weights : path to the file containing weight (last layer is a Linear Layer with 51 neurons)
        - num_classes : number of classes, for the weight file (rsp-aid-resnet-50-e300-ckpt.pth) we need to set num classes to 51
    Outputs
        - res50 : Resnet50 pretrained model
    """
    res50 = resnet50(num_classes = num_classes)
    res50_state = torch.load(path_to_weights) 
    res50.load_state_dict(res50_state["model"]) # we can add argument .load_state_dict( ... , strict = False) if the weights dont load properly, random weights will be intialised for the weights that do not work
    return res50

def freeze_all_but_last(model):
    """
    Desc : Freezes all layers but last layer (FC)
    Inputs
        - model : Model with pretrained weights
    Output
        - model : Model with forzen weights except last layer
    """

    # Freeze all layers of the model
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze Last layer which contains a fully connected layer (name of this layer is "fc" and its param names are "fc.weight", "fc.bias")
    model.fc.weight.requires_grad = True
    model.fc.bias.requires_grad = True

    # Note once frozen if you print the value requires_grad = True will not appear in frozen layers but will in unfrozen layer
    return model

def add_classification_layer(model, prev_hidden_size = 51, next_hidden_size = 43):
    """
    Desc : Adds a final classification layer (fully connected layer) to a pretrained model
    Inputs :
        - model : pretrained model
        - preve_hidden_size : Hidden units in the last layer of the pretrained model, default = 51 as BigEarthNet was trained on 51 labels 
        - neurons : Hidden units you wish to add as another layer to an existing model, default = 43 as BigeEarthnet has 43 labels
    Output
        - model : model with an additional layer.
    """

    # Add layer method 1
    # model = nn.Sequential(
    #     model,
    #     nn.Linear(prev_hidden_size, next_hidden_size)
    # )

    # Add layer method 2 : you can name it however you want this way
    model.fc1 = nn.Linear(prev_hidden_size, next_hidden_size)

    return model

if __name__ == "__main__":
    
    # Load pretrained weights 
    res50 = load_pretrained_weights()

    # Freeze all but last layer
    res50 = freeze_all_but_last(res50)

    # Add another FC layer for classifying 43 labels
    res50 = add_classification_layer(res50, prev_hidden_size = 51, next_hidden_size = 43)

    # for name, param in res50.named_parameters():
    #     print("-"*20)
    #     print(f"name : {name}")
    #     print(f"values : \n{param}")

    # print(res50)

    
