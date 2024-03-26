# ==============================================================================
# Training code for SimSiam
# Base line setting
#   optimizer : SGD, momentum = 0.9
#   learning rate :  base lr = 0.5 , lr = lr * BS/256
#   Cosine Decay : 0.0001
#   BS : 512
#   Projection Head : input_size = 2048, output_size = 2048, hidden_size = 2048
#   Prediction Head : Input_size = 2048, Output_size = 2048, hidden_size = 512

#todo : Need to incorporate cosine decay
#todo : lr = lr * BS/256 , where base_lr = 0.5
#! Cannot set batch size to 512 too big for memory
# ==============================================================================

# ---------------------------------- imports --------------------------------- #

import os
import sys
import argparse

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from lightly.loss import NegativeCosineSimilarity
from lightly.models.modules import SimSiamPredictionHead, SimSiamProjectionHead
from lightly.transforms import SimSiamTransform
from lightly.data import LightlyDataset

from utils import load_rsp_weights
import config


sys.path.append("RSP/Scene Recognition")
from models.resnet import resnet50

# ------------------------------ Argument Parse ------------------------------ #
parser = argparse.ArgumentParser()
parser.add_argument("-dfold", "--data_folder", type = str)
args = parser.parse_args()

# ------------------------------- SimSIam Model ------------------------------ #
class SimSiam(pl.LightningModule):
    def __init__(self):
        super().__init__()
        resnet = load_rsp_weights(
            resnet50, 
            path_to_weights = "pretrain_weights/rsp-aid-resnet-50-e300-ckpt.pth", 
            num_classes = 51
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.projection_head = SimSiamProjectionHead(2048,2048,2048) #These are the default values
        self.prediction_head = SimSiamPredictionHead(2048,128,2048) #These are deafult values, bottleneck created in original paper

        # Loss
        self.criterion = NegativeCosineSimilarity()

    def forward(self, X):
        f = self.backbone(X).flatten(start_dim = 1) #(b,3,256,256) -> (b,512,1,1) -> (b,512)
        z = self.projection_head(f) #(b,2048)
        p = self.prediction_head(z) #(b,2048)
        z = z.detach() #We stop the gradient to prevent collapse
        return z, p

    def training_step(self, batch, batch_idx):
        (X0,X1) = batch[0]
        z0,p0 = self.forward(X0) #(b,2048),(b,2048)
        z1,p1 = self.forward(X1) #(b,2048),(b,2048)
        # Compute loss
        loss = 0.5 * (self.criterion(z0,p1) + self.criterion(z1, p0))
        # Check std to see if model is collapsing
        #self.avg_loss = 0.0
        #avg_output_std = self.compute_std(p0, loss)
        # Store metrics 
        self.log("loss", loss)
        #self.log_dict({"loss" : loss, "avg_output_std" : avg_std_output})

    def compute_std(self,p0,loss):
        # CHeck if embeddings are collapsing
        #output = p0.copy_()#(b,2048)
        output = p0.detach()
        output = F.normalize(output, dim = 1)
        output_std = torch.std(output,0)#(2048,)
        output_std = output_std.mean() #() <- Its and Float value

        # Use  moving average to track the loss and standard deviation
        w = 0.9
        self.avg_loss = w * self.avg_loss + (1-w) * loss.item()
        avg_output_std = w * avg_output_std + (1 - w) * output_std.item()
        return avg_output_std

    def configure_optimizers(self):
        optim = torch.optim.SGD(self.parameters(), lr = 0.06)
        return optim

if __name__ == "__main__":
    # Path to folder containg images (.tif)
    data_root = "data/SSHSPH-RSMosaics-MY-v2.1/images"
    path_to_data = os.path.join(data_root, args.data_folder)
    transform = SimSiamTransform(input_size = 256)
    trainset = LightlyDataset(path_to_data, transform = transform)
    trainloader = DataLoader(trainset, batch_size = config.BATCH_SIZE, shuffle = True, drop_last = True)

    simsiam = SimSiam()

    trainer = pl.Trainer(
        default_root_dir = f"saved_weights/simsiam-is{config.INPUT_SIZE}-bs{config.BATCH_SIZE}-ep{config.MAX_EPOCHS}",
        devices = [0],
        accelerator = "gpu",
        max_epochs = config.MAX_EPOCHS
    )

    trainer.fit(simsiam, trainloader)

    # for (X0,X1),_,_ in trainloader:
    #     z0,p0 = simsiam.forward(X0) #(b,2048),(b,2048)
    #     z1,p1 = simsiam.forward(X1) #(b,2048),(b,2048)
    #     print(z0.shape, p0.shape)
    #     break