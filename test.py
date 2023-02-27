#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from train import train_bm_model, model_to_device
from model import BMModel

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
print("running bm_model_dilated_skip_conv")
writer = SummaryWriter("runs/bm_model_dilated_skip_conv")
model = BMModel(dilated_convs=True, large=False, think=False)
model = model_to_device(model)
train_bm_model(model, "models/bm_model_dilated_skip_conv.pth", summary_writer=writer)
writer.flush()