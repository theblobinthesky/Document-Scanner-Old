#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from train import train_bm_model, model_to_device
from torchinfo import summary
import bm_model
import model

class SigmoidOutputTestModule(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return torch.sigmoid(x)
    
    def loss(self, pred, dict, weight_metrics):
        return self.model.loss(pred, dict, weight_metrics)
    
    def input_and_label_from_dict(self, dict):
        return self.model.input_and_label_from_dict(dict)


print("running all tests")

# bm_model.use_relu = False
# bm_model.use_skip = True
# model.l2_weight = 0.0

# print()
# print("running no_relu")
# writer = SummaryWriter("runs/no_relu")
# model = bm_model.BMModel(True, False)
# model = model_to_device(model)
# train_bm_model(model, "models/no_relu.pth", summary_writer=writer)
# writer.flush()

bm_model.use_relu = True
bm_model.use_skip = True
model.l2_weight = 0.0

print()
print("running with_relu")
writer = SummaryWriter("runs/with_relu")
model = bm_model.BMModel(True, False)
model = model_to_device(model)
train_bm_model(model, "models/with_relu.pth", summary_writer=writer)
writer.flush()

bm_model.use_relu = False
bm_model.use_skip = True
model.l2_weight = 0.3

print()
print("running optimized_gru_skip_l2_30")
writer = SummaryWriter("runs/optimized_gru_skip_l2_30")
model = bm_model.BMModel(True, False)
model = model_to_device(model)
train_bm_model(model, "models/optimized_gru_skip_l2_30.pth", summary_writer=writer)
writer.flush()

model.l2_weight = 0.6

print()
print("running optimized_gru_skip_l2_60")
writer = SummaryWriter("runs/optimized_gru_skip_l2_60")
model = bm_model.BMModel(True, False)
model = model_to_device(model)
train_bm_model(model, "models/optimized_gru_skip_l2_60.pth", summary_writer=writer)
writer.flush()

model.l2_weight = 0.9

print()
print("running optimized_gru_skip_l2_90")
writer = SummaryWriter("runs/optimized_gru_skip_l2_90")
model = bm_model.BMModel(True, False)
model = model_to_device(model)
train_bm_model(model, "models/optimized_gru_skip_l2_90.pth", summary_writer=writer)
writer.flush()