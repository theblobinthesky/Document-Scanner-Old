#!/usr/bin/python3
from model import Model, BMModel, save_model, load_model
from data import prepare_datasets
from torchvision.transforms import Resize
import torch
from torchinfo import summary
import datetime
from itertools import cycle
from tqdm import tqdm

lr = 1e-3
steps_per_epoch = 20
batch_size = 8
device = torch.device('cuda')

mask_epochs = 800
wc_epochs = 2000
bm_epochs = 2000

mask_time_in_hours = 2.0
wc_time_in_hours = 0 # 2.5
bm_time_in_hours = 0 # 1.0

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
        

def model_to_device(model):
    return model.to(device)


def train_model(mode, model, epochs, time_in_hours, trainds_iter, summary_writer):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=5)

    start_time = datetime.datetime.now()

    pbar = tqdm(total=epochs)
    pbar.set_description("initializing training and running first epoch")
    
    for epoch in range(epochs):
        train_loss = 0.0

        for i in range(steps_per_epoch):
            dict = next(trainds_iter)

            for key in dict.keys():
                dict[key] = dict[key].to(device)
                
            if mode == 0:
                img = dict["img"]
                label = dict["lines"][:, [0, 2]]
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

            train_loss += loss.detach().cpu().item()

            # print(f"epoch {epoch}/{epochs} step {i + 1}/{steps_per_epoch}. training loss: {loss.item():.4}")

        train_loss /= steps_per_epoch
        valid_loss = 0.0
        # valid_loss = model.eval_loss_on_ds(valid_ds)

        scheduler.step(train_loss)

        now = datetime.datetime.now()
        hours_passed = (now - start_time).seconds / (60.0 * 60.0)

        summary_writer.add_scalar("Loss/train", train_loss, epoch)

        pbar.update(1)
        pbar.set_description(f"epoch {epoch + 1}/{epochs} completed with {hours_passed:.4f}h passed. training loss: {train_loss:.4}, validation loss: {valid_loss:.4}, learning rate: {optim.param_groups[-1]['lr']}")

        if hours_passed > time_in_hours:
            pbar.close()
            print(f"hour limit of {time_in_hours:.4f}h passed")
            break

    pbar.close()


transform = Resize((128, 128))

def train_pre_model(model, model_path, summary_writer):
    train_ds, valid_ds, test_ds = prepare_datasets("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", {"img": "png", "lines": "png"}, [
        ([("img", "img/1"), ("lines", "lines/1")], 5000),
        ([("img", "img/2"), ("lines", "lines/2")], 5000),
        #([("img", "img/3"), ("lines", "lines/3")], 5000)
    ], valid_perc=0.1, test_perc=0.1, batch_size=batch_size, transform=transform)

    trainds_iter = iter(cycle(train_ds))

    train_model(0, model, mask_epochs, mask_time_in_hours, trainds_iter, summary_writer)

    save_model(model, model_path)


def train_wc_model(model, model_path, summary_writer):
    train_ds, valid_ds, test_ds = prepare_datasets([
        ("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", [("img_masked/1", "png"), ("lines/1", "png")], "wc/1", "exr", 20000)
    ], valid_perc=0.1, test_perc=0.1, batch_size=batch_size, transform=transform)

    trainds_iter = iter(cycle(train_ds))

    train_model(1, model, wc_epochs, wc_time_in_hours, trainds_iter, summary_writer)

    save_model(model, model_path)


def train_bm_model(model, model_path, summary_writer):
    train_ds, valid_ds, test_ds = prepare_datasets([
        ("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", [("wc/1", "exr")], "bm/1exr", "exr", 20000)
    ], valid_perc=0.1, test_perc=0.1, batch_size=batch_size, transform=transform)

    trainds_iter = iter(cycle(train_ds))

    train_model(2, model, bm_epochs, bm_time_in_hours, trainds_iter, summary_writer)

    save_model(model, model_path)


if __name__ == "__main__":
    model = load_model("model unet transformer.pth")
    model.to(device)
    summary(model.pre_model, input_size=(1, 3, 128, 128), device=device)
