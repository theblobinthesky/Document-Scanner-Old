#!/usr/bin/python3
from model import Model, save_model, load_model, binarize
from data import prepare_datasets
import torch
from torchinfo import summary
import datetime
from itertools import cycle

lr = 1e-3
steps_per_epoch = 20
device = torch.device('cuda')

mask_epochs = 200
uv_epochs = 200

mask_time_in_hours = 1.0
uv_time_in_hours = 2.0

train_ds, valid_ds, test_ds = prepare_datasets([
    ("/media/shared/Projekte/Scanner/datasets/Doc3d", "img", "uv_exr", "png", "exr", 20000)
], valid_perc=0.1, test_perc=0.1, batch_size=16)

trainds_iter = iter(cycle(train_ds))

model = Model() # load_model("model.pth")
model.to(device)

summary(model, input_size=(1, 3, 128, 128), device=device)


def train_model(is_mask, model, epochs, time_in_hours):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=5)

    start_time = datetime.datetime.now()

    for epoch in range(epochs):
        train_loss = 0.0

        for i in range(steps_per_epoch):
            img, label = next(trainds_iter)
            img = img.to(device)
            label = label.to(device)

            if is_mask:
                label = label[:, 2, :, :].unsqueeze(axis=1)
            else:
                mask = label[:, 2, :, :].unsqueeze(axis=1)
                mask = binarize(mask)
                label = label[:, 0:2, :, :]
                img = img * mask

            optim.zero_grad(set_to_none=True)
            
            pred = model(img)
            loss = model.loss(pred, label)
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


print("training mask model")
train_model(True, model.mask_model, mask_epochs, mask_time_in_hours)

save_model(model, "model.pth")

print()
print("training uv model")
train_model(False, model.uv_model, uv_epochs, uv_time_in_hours)

test_loss = 0.0
# test_loss = model.eval_loss_on_ds(test_ds)
print(f"test loss: {test_loss:.4}")

save_model(model, "model.pth")