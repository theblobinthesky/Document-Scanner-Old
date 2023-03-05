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

# print()
# print("running color_jitter")
# writer = SummaryWriter("runs/color_jitter")
# model = bm_model.BMModel(True, False)
# model = model_to_device(model)
# train_bm_model(model, "models/color_jitter.pth", summary_writer=writer)
# writer.flush()

print()
print("running color_jitter_and_upscaling")
writer = SummaryWriter("runs/color_jitter_and_upscaling")
model = bm_model.BMModel(True, True)
model = model_to_device(model)
train_bm_model(model, "models/color_jitter_and_upscaling.pth", summary_writer=writer)
writer.flush()