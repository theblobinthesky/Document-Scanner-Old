#!/usr/bin/python3
from model import Model, loss_function, save_model, load_model
from data import prepare_datasets
import torch
from torchinfo import summary
import datetime
from itertools import cycle

lr = 1e-3
epochs = 200
steps_per_epoch = 20
device = torch.device('cuda')
time_in_hours = 1.5 / 60.0

train_ds, valid_ds, test_ds = prepare_datasets([
    ("/media/shared/Projekte/Scanner/datasets/Doc3d", "img", "uv_exr", "png", "exr", 5000)
], valid_perc=0.1, test_perc=0.1, batch_size=16, device=device)

trainds_iter = iter(cycle(train_ds))

model = load_model("model tr.pth")
model.to(device)

summary(model, input_size=(1, 3, 128, 128), device=device)

optim = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=5)

start_time = datetime.datetime.now()

for epoch in range(epochs):
    train_loss = 0.0

    for i in range(steps_per_epoch):
        img, uv_label = next(trainds_iter)
        img = img.to(device)
        uv_label = uv_label.to(device)

        optim.zero_grad(set_to_none=True)
        
        uv_pred = model(img)
        loss = loss_function(uv_pred, uv_label)
        loss.backward()

        optim.step()

        train_loss += loss.item()

        # print(f"epoch {epoch}/{epochs} step {i + 1}/{steps_per_epoch}. training loss: {loss.item():.4}")

    train_loss /= steps_per_epoch
    valid_loss = 0.0
    # valid_loss = model.eval_loss_on_ds(valid_ds)

    scheduler.step(train_loss)

    now = datetime.datetime.now()
    hours_passed = (now - start_time).seconds / (60.0 * 60.0)

    print(f"epoch {epoch + 1}/{epochs} completed with {hours_passed:.4f}h passed. training loss: {train_loss:.4}, validation loss: {valid_loss:.4}, learning rate: {optim.param_groups[-1]['lr']}")

    if hours_passed > time_in_hours:
        print(f"hour limit of {time_in_hours:.4f}h passed")
        break


test_loss = 0.0
# test_loss = model.eval_loss_on_ds(test_ds)
print(f"test loss: {test_loss:.4}")

save_model(model, "model tr.pth")