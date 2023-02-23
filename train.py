#!/usr/bin/python3
from model import Model, BMModel, save_model, load_model, binarize
from data import prepare_datasets
from benchmark import benchmark_eval
from torchvision.transforms import Resize
import torch
# from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
import datetime
from itertools import cycle

lr = 1e-3
steps_per_epoch = 20
batch_size = 8
device = torch.device('cuda')

mask_epochs = 2000
wc_epochs = 2000
bm_epochs = 2000

mask_time_in_hours = 0
wc_time_in_hours = 2.5
bm_time_in_hours = 1.0

model = load_model("model.pth")
model.to(device)
# writer = SummaryWriter()

summary(model, input_size=(1, 3, 128, 128), device=device)


def train_model(mode, model, epochs, time_in_hours, trainds_iter):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=5)

    start_time = datetime.datetime.now()

    for epoch in range(epochs):
        train_loss = 0.0

        for i in range(steps_per_epoch):
            img, label = next(trainds_iter)
            img = img.to(device)
            label = label.to(device)

            if mode == 0:
                label = label[:, 2].unsqueeze(axis=1)
            elif mode == 1:
                img[:, 0:3] *= img[:, 5].unsqueeze(axis=1) # todo: this might be a bug
                img = img[:, [0, 1, 2, 3]]
            elif mode == 2:
                label = label[:, :2] / 448.0

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

        # writer.add_scalar("Loss/train", train_loss, epoch)
        # writer.add_scalar("Loss/validation", valid_loss, epoch)

        print(f"epoch {epoch + 1}/{epochs} completed with {hours_passed:.4f}h passed. training loss: {train_loss:.4}, validation loss: {valid_loss:.4}, learning rate: {optim.param_groups[-1]['lr']}")

        if hours_passed > time_in_hours:
            print(f"hour limit of {time_in_hours:.4f}h passed")
            break
        # elif keyboard.is_pressed('v'):
        #     model.train(False)
        #     benchmark_eval(model)
        #     save_model(model, "model.pth")
        #     model.train(True)
        # elif keyboard.is_pressed('q'):
        #     save_model(model, "model.pth")

        #     exit()


transform = Resize((128, 128))

def train_pre_model():
    train_ds, valid_ds, test_ds = prepare_datasets([
        ("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", [("img/1", "png")], "wc/1", "exr", 20000)
    ], valid_perc=0.1, test_perc=0.1, batch_size=batch_size, transform=transform)

    trainds_iter = iter(cycle(train_ds))

    print("training mask model")
    train_model(0, model.pre_model, mask_epochs, mask_time_in_hours, trainds_iter)

    save_model(model, "model.pth")

#train_pre_model()


def train_wc_model():
    train_ds, valid_ds, test_ds = prepare_datasets([
        ("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", [("img_masked/1", "png"), ("lines/1", "png")], "wc/1", "exr", 20000)
    ], valid_perc=0.1, test_perc=0.1, batch_size=batch_size, transform=transform)

    trainds_iter = iter(cycle(train_ds))

    print()
    print("training wc model")
    train_model(1, model.wc_model, wc_epochs, wc_time_in_hours, trainds_iter)

    save_model(model, "model.pth")

train_wc_model()


def train_bm_model():
    train_ds, valid_ds, test_ds = prepare_datasets([
        ("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", [("wc/1", "exr")], "bm/1exr", "exr", 20000)
    ], valid_perc=0.1, test_perc=0.1, batch_size=batch_size, transform=transform)

    trainds_iter = iter(cycle(train_ds))

    print()
    print("training bm model")
    train_model(2, model.bm_model, bm_epochs, bm_time_in_hours, trainds_iter)

train_bm_model()


test_loss = 0.0
# test_loss = model.eval_loss_on_ds(test_ds)
print(f"test loss: {test_loss:.4}")

save_model(model, "model.pth")