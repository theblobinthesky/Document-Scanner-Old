#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from train import train_bm_model, model_to_device
from model import BMModel
from torchinfo import summary

class SigmoidOutputTestModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
    
    def loss(self, pred, dict):
        return self.model.loss(pred, dict)
    
    def input_from_dict(self, dict):
        return self.model.input_from_dict(dict)


print("running all tests")

import model

model.cc_loss_enabled = True

print()
print("running test")
writer = SummaryWriter("runs/test")
model = BMModel(True, False, False)
summary(model, input_size=(1, 3, 128, 128))
model = model_to_device(model)
train_bm_model(model, "models/test.pth", summary_writer=writer)
writer.flush()