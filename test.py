#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from train import train_pre_model, model_to_device
from model import PreModel

class SigmoidOutputTestModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
    
    def loss(self, pred, label):
        return self.model.loss(pred, label)
    
    def x_and_y_from_dict(self, dict):
        return self.model.x_and_y_from_dict(dict)


print("running all tests")


print()
print("running pre_model_baseline")
writer = SummaryWriter("runs/pre_model_baseline")
model = PreModel(focal_loss=False)
model.load_state_dict(torch.load("models/pre_model_baseline.pth"))
unet = model_to_device(model)
train_pre_model(unet, "models/pre_model_baseline.pth", summary_writer=writer)
writer.flush()



print()
print("running pre_model_focal_loss")
writer = SummaryWriter("runs/pre_model_focal_loss")
model = PreModel(focal_loss=True)
model.load_state_dict(torch.load("models/pre_model_focal_loss.pth"))
unet = model_to_device(PreModel(focal_loss=True))
train_pre_model(unet, "models/pre_model_focal_loss.pth", summary_writer=writer)
writer.flush()