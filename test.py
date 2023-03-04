#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from train import train_bm_model, model_to_device
from torchinfo import summary
import bm_model


class SigmoidOutputTestModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
    
    def loss(self, pred, dict):
        return self.model.loss(pred, dict)
    
    def input_and_label_from_dict(self, dict):
        return self.model.input_and_label_from_dict(dict)


print("running all tests")

print()
print("running bm_progressive_baseline")
writer = SummaryWriter("runs/bm_progressive_baseline")
model = bm_model.BMModel(True)
model = model_to_device(model)
train_bm_model(model, "models/bm_progressive_baseline.pth", summary_writer=writer)
writer.flush()