#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from train import train_pre_model, model_to_device
from model import UNet
from seg_model import UNetTransformer, UNetDilatedConv

class SigmoidOutputTestModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
    
    def loss(self, pred, label):
        return self.model.loss(pred, label)


print("running all tests")

print()
print("running unet baseline model")

writer = SummaryWriter("runs/pre_model_unet_baseline")
unet = model_to_device(SigmoidOutputTestModule(UNet(3, 2, blocks=2)))
train_pre_model(unet, "models/pre_model_unet_baseline.pth", summary_writer=writer)
writer.flush()


print()
print("running unet transformer model")

writer = SummaryWriter("runs/pre_model_unet_transformer")
unet = model_to_device(SigmoidOutputTestModule(UNetTransformer(3, 2)))
train_pre_model(unet, "models/pre_model_unet_transformer.pth", summary_writer=writer)
writer.flush()


print()
print("running unet dilated convolution model")

writer = SummaryWriter("runs/pre_model_unet_dilated_convs")
unet = model_to_device(SigmoidOutputTestModule(UNetDilatedConv(3, 2)))
train_pre_model(unet, "models/pre_model_unet_dilated_convs.pth", summary_writer=writer)
writer.flush()