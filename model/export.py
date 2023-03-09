#!/usr/bin/python3
import torch
import onnx
import os
import shutil
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from onnx_tf.backend import prepare
import tensorflow as tf

from seg_model import PreModel

dummy_input = torch.randn(1, 3, 128, 128)

def export(model, path_without_extension):
    print(f"Exporting {path_without_extension}")

    onnx_path = path_without_extension + ".onnx"
    tf_path = path_without_extension + "_tf"
    tflite_path = path_without_extension + ".tflite"

    torch.onnx.export(model, dummy_input, onnx_path, opset_version=12, input_names=['input'], output_names=['output'])

    onnx_model = onnx.load(onnx_path)
    tf_rep = prepare(onnx_model)
    tf_rep.export_graph(tf_path)

    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)
    tflite_model = converter.convert()

    with open(tflite_path, "wb") as f:
        f.write(tflite_model)

    os.remove(onnx_path)
    shutil.rmtree(tf_path)


model = PreModel()
model.load_state_dict(torch.load("models/main_seg_model.pth"))
export(model, "exports/seg_model")