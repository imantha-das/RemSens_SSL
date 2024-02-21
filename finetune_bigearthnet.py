import os
import sys
import collections
import json
import pandas as pd
from termcolor import colored
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize
import pytorch_lightning as pl
from bigearthnet_utils.ben_utils import BigEarthNet 

# Load path to RSP Scene Recognition repository in order to load pretrained model
sys.path.append("/home/imantha/workspace/RemSens_SSL/RSP/Scene Recognition")

#from models import build_model
from models.resnet import resnet50

def load_model_and_weights(path_to_weights = "pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth", num_classes = 51):
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

class BEN_Classifier(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()

        AID_CLASSES = 51
        PATH_TO_WEIGHTS = "pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth"
        # load pretrained model and pretrained weights
        self.model = load_model_and_weights(path_to_weights = PATH_TO_WEIGHTS, num_classes = AID_CLASSES)
        # freeze all but the last layer which is an FC layer, backbone which is resnet50 gets its weights frozen
        self.model = freeze_all_but_last(self.model)
        # Change last layer which is named fc to have 43 ouptuts instead of 51
        self.model.fc = nn.Linear(
            in_features = self.model.fc.in_features,
            out_features = num_classes
        )

    def training_step(self, batch, batch_idx):
        """Trains models and returns the loss"""
        X,y = batch
        yhat = self.model(X)
        #print(colored(f"yhat.shape : {yhat.shape} , y.shape : {y.shape}", "red"))
        loss = F.binary_cross_entropy_with_logits(yhat, y, reduction = "mean")
        self.log("train_loss", loss)
        return loss 

    def validation_step(self, batch, batch_idx):
        X, y = batch
        yhat = self.model(X)
        loss = F.binary_cross_entropy_with_logits(yhat, y, reduction = "mean")
        self.log("valid_loss", loss)
        return loss

    def configure_optimizers(self):
        """Returns the optimizer used for backpropogation"""
        return optim.AdamW(self.model.parameters(), lr = 0.0001)




if __name__ == "__main__":
    
    # ----------------------------- Load BigEarthNet ----------------------------- #
    # Load csv file containing paths to images and label encordings
    df_paths_enc = pd.read_csv("bigearthnet_utils/ben_datapaths_and_labels_v4.csv")
    df_paths_enc.drop("Unnamed: 0", axis =1 , errors = "ignore", inplace = True)
    
    # Load json file containing mean and std for normaizer
    with open("bigearthnet_utils/ben_metadata.json") as f:
        metadata = json.load(f)

    mean_norm = metadata["train_mean"]
    std_norm = metadata["train_std"]

    # Torch Dataset for BigEarthNet
    ben = BigEarthNet(paths_and_labs = df_paths_enc, transform = Compose([Normalize(mean_norm, std_norm)]))

    # Split dataset into train, valid, test
    train_size = int(0.2 * len(ben)) # 118065 images
    test_size = len(ben) - train_size # 472261 images
    train_data, test_data = torch.utils.data.random_split(ben, [train_size, test_size])

    valid_size = int(train_size/2) # 59032 images
    train_size = train_size - valid_size # 59033 images

    train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, valid_size])

    # Load datasets into Dataloaders
    train_loader = DataLoader(train_data, batch_size = 64, shuffle = True)
    valid_loader = DataLoader(valid_data,)

    # ------------------------ Pytorch Lightning Training ------------------------ #

    classifier = BEN_Classifier(num_classes = 43) 
    trainer = pl.Trainer(accelerator = "gpu", enable_progress_bar = True, min_epochs = 1, max_epochs = 10)
    trainer.fit(classifier, train_loader, valid_loader)

    # print(classifier.model)

    # for name, param in classifier.model.named_parameters():
    #     print("-"*20)
    #     print(f"name : {name}")
    #     print(f"values : \n{param}")

    # ----------------------------- Pytorch training ----------------------------- #

    # AID_CLASSES = 51
    # NUM_CLASSES = 43
    # PATH_TO_WEIGHTS = "pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth"
    # # load pretrained model and pretrained weights
    # model = load_model_and_weights(path_to_weights = PATH_TO_WEIGHTS, num_classes = AID_CLASSES)
    # # freeze all but the last layer which is an FC layer, backbone which is resnet50 gets its weights frozen
    # model = freeze_all_but_last(model)
    # # Change last layer which is named fc to have 43 ouptuts instead of 51
    # model.fc = nn.Linear(
    #     in_features = model.fc.in_features,
    #     out_features = NUM_CLASSES
    # )

    # EPOCHS = 3
    # LR = 0.0001
    # device = torch.device("cuda") if torch.cuda.is_available else torch.device("cpu") 
    # criterion = nn.BCEWithLogitsLoss()
    # optimizer = optim.AdamW(model.parameters(), lr = LR)

    # for e in range(EPOCHS):
    #     train_loss = []
    #     for batch in train_loader:
    #         X, y = batch
    #         #X = X.to(device) ; y = y.to(device)
    #         yhat = model.forward(X)          
    #         loss = criterion(yhat, y)

    #         train_loss.append(loss.item())
    #         print(f"Training loss : {loss.item()}")






    



