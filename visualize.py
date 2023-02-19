from model import load_model
from data import prepare_datasets
import matplotlib.pyplot as plt

model = load_model("model.pth")
model.eval()

