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
from model import Model, device
from seg_model import ContourModel
from bm_model import BMModel

name = "heatmap_10"
model_type = Model.CONTOUR

# todo: remove this later bc. it needs to be all 4 channels to be fast for mobile inference
if model_type == Model.CONTOUR:
    size = (64, 64)
    dummy_input = torch.randn(1, 4, *size, device=device)
elif model_type == Model.BM:
    size = (32, 32)
    dummy_input = torch.randn(1, 3, *size, device=device)

def export(model, tflite_path):
    print(f"Exporting tflite file to {tflite_path}")

    onxx_file = io.BytesIO()

    global model_type
    if model_type == Model.CONTOUR:
        torch.onnx.export(model, dummy_input, onxx_file, opset_version=9, input_names=["input"], output_names=["output"])
        # opset_version=9 is required since Upsampling is replaced by Resize in more recent versions.
        # Also there seem to be other nontrivial changes.

        onnx_model = onnx.load_model_from_string(onxx_file.getvalue())
        keras_model = onnx_to_keras(onnx_model, ["input"], verbose=False, name_policy='renumerate', change_ordering=True)
    elif model_type == Model.BM:
        torch.onnx.export(model, dummy_input, onxx_file, opset_version=16, input_names=["input"], output_names=["output"])

        onnx_model = onnx.load_model_from_string(onxx_file.getvalue())
        keras_model = onnx_to_keras(onnx_model, ["input"], verbose=False, name_policy='renumerate')


    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.inference_input_type = tf.float32
    converter.inference_output_type = tf.float32
    tflite_model = converter.convert()

    print()
    tf.lite.experimental.Analyzer.analyze(model_content=tflite_model, gpu_compatibility=True)

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

if model_type == Model.SEG:
    print("SEG model export is not supported yet.")
    exit()
elif model_type == Model.CONTOUR:
    model = ContourModel()
elif model_type == Model.BM:
    model = BMModel(False, False)

model.load_state_dict(torch.load(f"models/{name}.pth"))
model = model.to(device=device)
export(model, f"exports/{name}.tflite")
exit()

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

output = interpreter.get_tensor(output_details[1]['index'])
output = output[0]

import matplotlib.pyplot as plt
plt.imshow(output)
plt.show()