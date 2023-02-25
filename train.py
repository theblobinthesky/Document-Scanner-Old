#!/usr/bin/python3
from model import Model, save_model, load_model, eval_loss_on_batches
from data import prepare_datasets
from torchvision.transforms import Resize
import torch
from torchinfo import summary
import datetime
from itertools import cycle
from tqdm import tqdm

lr = 1e-3
steps_per_epoch = 40
batch_size = 8
device = torch.device('cuda')

mask_epochs = 200
wc_epochs = 2000
bm_epochs = 2000
min_learning_rate_before_early_termination = 1e-6
lr_plateau_patience = 3
valid_batch_count = 16
valid_eval_every = 4

mask_time_in_hours = 2.0
wc_time_in_hours = 0 # 2.5
bm_time_in_hours = 0 # 1.0

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
        

def model_to_device(model):
    return model.to(device)


def train_model(model, epochs, time_in_hours, trainds_iter, valid_iter, summary_writer):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=lr_plateau_patience)

    start_time = datetime.datetime.now()

    pbar = tqdm(total=epochs)
    pbar.set_description("initializing training and running first epoch")
    
    for epoch in range(epochs):
        train_loss = 0.0

        for _ in range(steps_per_epoch):
            dict = next(trainds_iter)

            for key in dict.keys():
                dict[key] = dict[key].to(device)
                
            x, y = model.x_and_y_from_dict(dict)
            # elif mode == 1:
            #     img[:, 0:3] *= img[:, 5].unsqueeze(axis=1) # todo: this might be a bug
            #     img = img[:, [0, 1, 2, 3]]
            # elif mode == 2:
            #     label = label[:, :2] / 448.0

            optim.zero_grad(set_to_none=True)

            pred = model(x)
            loss = model.loss(pred, y)
            loss.backward()

            optim.step()

            train_loss += loss.detach().cpu().item()

        train_loss /= steps_per_epoch

        if epoch % valid_eval_every == 0:
            valid_loss = eval_loss_on_batches(model, valid_iter, valid_batch_count, device)
            summary_writer.add_scalar("Loss/valid", valid_loss, epoch)

        scheduler.step(train_loss)

        now = datetime.datetime.now()
        hours_passed = (now - start_time).seconds / (60.0 * 60.0)
        learning_rate = optim.param_groups[-1]['lr']

        summary_writer.add_scalar("Loss/train", train_loss, epoch)

        pbar.update(1)
        pbar.set_description(f"epoch {epoch + 1}/{epochs} completed with {hours_passed:.4f}h passed. training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}, learning rate: {learning_rate:.4f}")

        if hours_passed > time_in_hours:
            pbar.close()
            print(f"hour limit of {time_in_hours:.4f}h passed")
            break

        if learning_rate < min_learning_rate_before_early_termination:
            pbar.close()
            print(f"learning rate {learning_rate:.4f} is smaller than {min_learning_rate_before_early_termination:.4f}. no further learning progress can be made")
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
    valid_iter = iter(cycle(valid_ds))

    train_model(model, mask_epochs, mask_time_in_hours, trainds_iter, valid_iter, summary_writer)

    save_model(model, model_path)


def train_wc_model(model, model_path, summary_writer):
    train_ds, valid_ds, test_ds = prepare_datasets([
        ("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", [("img_masked/1", "png"), ("lines/1", "png")], "wc/1", "exr", 20000)
    ], valid_perc=0.1, test_perc=0.1, batch_size=batch_size, transform=transform)

    trainds_iter = iter(cycle(train_ds))
    valid_iter = iter(cycle(valid_ds))

    train_model(model, wc_epochs, wc_time_in_hours, trainds_iter, valid_iter, summary_writer)

    save_model(model, model_path)


def train_bm_model(model, model_path, summary_writer):
    train_ds, valid_ds, test_ds = prepare_datasets([
        ("/media/shared/Projekte/DocumentScanner/datasets/Doc3d", [("wc/1", "exr")], "bm/1exr", "exr", 20000)
    ], valid_perc=0.1, test_perc=0.1, batch_size=batch_size, transform=transform)

    trainds_iter = iter(cycle(train_ds))
    valid_iter = iter(cycle(valid_ds))

    train_model(model, bm_epochs, bm_time_in_hours, trainds_iter, valid_iter, summary_writer)

    save_model(model, model_path)


if __name__ == "__main__":
    model = load_model("model unet transformer.pth")
    model.to(device)
    summary(model.pre_model, input_size=(1, 3, 128, 128), device=device)
