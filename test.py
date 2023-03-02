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
    
    def loss(self, pred, dict):
        return self.model.loss(pred, dict)
    
    def x_and_y_from_dict(self, dict):
        return self.model.x_and_y_from_dict(dict)

import bm_model

print("running all tests")

bm_model.max_iters = 1

print()
print("running bm_progressive")
writer = SummaryWriter("runs/bm_progressive")
model = BMModel(True, False, False)
model = model_to_device(model)
train_bm_model(model, "models/bm_progressive.pth", summary_writer=writer)
writer.flush()


bm_model.max_iters = 4

print()
print("running bm_progressive_iters")
writer = SummaryWriter("runs/bm_progressive_iters")
model = BMModel(True, False, False)
model = model_to_device(model)
train_bm_model(model, "models/bm_progressive_iters.pth", summary_writer=writer)
writer.flush()
