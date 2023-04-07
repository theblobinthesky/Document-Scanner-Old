#!/usr/bin/python3
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch
from train import train_model, model_to_device
from torchinfo import summary
import bm_model
import seg_model
from model import Model

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

def test_contour_model(name):
    print(f"testing {name}...")

    writer = SummaryWriter(f"runs/{name}")
    model = seg_model.ContourModel()
    model = model_to_device(model)
    train_model(model, f"models/{name}.pth", Model.CONTOUR, summary_writer=writer)
    writer.flush()

seg_model.pos_weight = 25
test_contour_model("heatmap_test_weight_25")

seg_model.pos_weight = 50
test_contour_model("heatmap_test_weight_50")