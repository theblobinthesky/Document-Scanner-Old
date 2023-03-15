#!/usr/bin/python3
import torch
import torch.nn as nn
import onnx
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
# Its very important to note that as of 2023 the onnx2keras repository had three years of development done
# without a release on github or the package on pip. onnx2keras must be directly cloned from github otherwise 
# the library is not necessarily compatible with modern tensorflow. 
from onnx2keras import onnx_to_keras
import io

size = (64, 64)
name = "seg_model_finetuning"

dummy_input = torch.randn(1, 4, size[0], size[1])

def export(model, tflite_path):
    print(f"Exporting tflite file to {tflite_path}")

    onxx_file = io.BytesIO()
    torch.onnx.export(model, dummy_input, onxx_file, opset_version=9, input_names=['input'], output_names=['output'])
    # opset_version=9 is required since Upsampling is replaced by Resize in more recent versions.
    # Also there seem to be some other nontrivial changes.

    onnx_model = onnx.load_model_from_string(onxx_file.getvalue())
    keras_model = onnx_to_keras(onnx_model, ["input"], verbose=False, name_policy='renumerate', change_ordering=True)

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

from seg_model import PreModel, binarize

class PreModelWrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = PreModel()
        self.model.load_state_dict(torch.load(f"models/{name}.pth"))

    def forward(self, x):
        x = self.model(x)

        return x


model = PreModelWrapper()
export(model, f"exports/{name}.tflite")

import cv2
import numpy as np
np_features = cv2.imread("/media/shared/Projekte/DocumentScanner/datasets/Doc3d/img/2/996_6-pr_Page_025-bgI0001.png")
np_features = cv2.resize(np_features, size)

np_features = np_features.astype("float32") / 255.0

h, w, _ = np_features.shape
padding = np.full((h, w, 1), 1.0, dtype=np.float32)
np_features = np.concatenate([np_features, padding], axis=-1)

interpreter = tf.lite.Interpreter(model_path=f"exports/{name}.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.allocate_tensors()

print("TENSOR-DETAILS FROM HERE:")
for ten_details in interpreter.get_tensor_details():
    name = ten_details["name"]
    if name != "": print(name)

print("------------------------")

np_features = np.expand_dims(np_features, axis=0)

interpreter.set_tensor(input_details[0]['index'], np_features)
interpreter.invoke()

output = interpreter.get_tensor(output_details[0]['index'])
output = output[0]

import matplotlib.pyplot as plt
plt.imshow(output)
plt.show()