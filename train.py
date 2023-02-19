#!/usr/bin/python3
from model import Model, loss_function
from data import prepare_datasets
import torch

lr = 1e-3
bsize = 64
epochs = 4
steps_per_epoch = 100

dsiter = prepare_datasets([
    ("/media/shared/Projekte/Scanner/datasets/Doc3d", "img", "uv_exr", "png", "exr", 10)
])

model = Model()

optim = torch.optim.Adam(model.parameters(), lr=lr)

def eval_valid_loss():
    return 0.0

for epoch in range(epochs):
    train_loss = 0.0

    for i in range(steps_per_epoch):
        data = next(dsiter)
        print(data)
        exit()(img, uv_label)

        optim.zero_grad(set_to_none=None)
        
        uv_pred = model(input)
        loss = loss_function(uv_pred, uv_label)
        loss.backward()

        train_loss += loss.item()

        optim.step()

    train_loss /= steps_per_epoch
    valid_loss = eval_valid_loss()

    print(f"epoch {epoch}/{epochs} completed. training loss: {train_loss}, validation loss: {valid_loss}")