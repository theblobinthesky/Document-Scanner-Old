#!/usr/bin/python3
from model import save_model, load_model, eval_loss_on_batches, eval_loss_and_metrics_on_batches, count_params
from data import load_datasets, prepare_pre_dataset, prepare_bm_dataset
import torch
from torchinfo import summary
import datetime
from itertools import cycle
from tqdm import tqdm
from benchmark import benchmark_plt

lr = 1e-3
steps_per_epoch = 40
batch_size = 12
device = torch.device("cuda")

mask_epochs = 300
wc_epochs = 2000
bm_epochs = 300
min_learning_rate_before_early_termination = 1e-7
lr_plateau_patience = 3
valid_batch_count = 16
valid_eval_every = 4
lam = 0.85

mask_time_in_hours = 2.0
wc_time_in_hours = 0 # 2.5
bm_time_in_hours = 8.0


def cycle(iterable):
    while True:
        for x in iterable:
            yield x
        

def model_to_device(model):
    return model.to(device)


def train_model(model, model_path, epochs, time_in_hours, ds, is_rnn, summary_writer):
    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, factor=0.1, patience=lr_plateau_patience)

    start_time = datetime.datetime.now()

    train_ds, valid_ds, test_ds = load_datasets(*ds, batch_size)
    trainds_iter = iter(cycle(train_ds))
    valid_iter = iter(cycle(valid_ds))
    test_iter = iter(test_ds)

    pbar = tqdm(total=epochs)
    pbar.set_description("initializing training and running first epoch")
    
    for epoch in range(epochs):
        train_loss = 0.0

        for _ in range(steps_per_epoch):
            dict = next(trainds_iter)

            for key in dict.keys():
                dict[key] = dict[key].to(device)
                
            x, _ = model.input_and_label_from_dict(dict)

            optim.zero_grad(set_to_none=True)

            if is_rnn:
                preds = model.forward_all(x)
                
                loss = 0.0

                for i, pred in enumerate(preds):
                    fac = lam ** (len(preds) - 1 - i)
                    loss += fac * model.loss(pred, dict)

                loss /= float(len(preds))
            else:
                pred = model(x)
                loss = model.loss(pred, dict)
            
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
        pbar.set_description(f"epoch {epoch + 1}/{epochs} completed with {hours_passed:.4f}h passed. training loss: {train_loss:.4f}, validation loss: {valid_loss:.4f}, learning rate: {learning_rate}")

        if hours_passed > time_in_hours:
            pbar.close()
            print(f"hour limit of {time_in_hours:.4f}h passed")
            break

        if learning_rate < min_learning_rate_before_early_termination:
            pbar.close()
            print(f"learning rate {learning_rate} is smaller than {min_learning_rate_before_early_termination}. no further learning progress can be made")
            break

    save_model(model, model_path)

    pbar.close()

    model.set_train(False)

    test_loss, metrics = eval_loss_and_metrics_on_batches(model, test_iter, batch_size, device)

    hparams = {"parameter_count": count_params(model)}

    metric_dict = {"Loss/test": test_loss}
    for (name, item) in metrics:
        metric_dict[name] = item

    summary_writer.add_hparams(hparams, metric_dict)

    plt = benchmark_plt(model, ds, is_rnn)
    summary_writer.add_figure("benchmark", plt)
    
    print(f"test loss: {test_loss:.4f}")


def train_pre_model(model, model_path, summary_writer):
    ds = prepare_pre_dataset()
    train_model(model, model_path, mask_epochs, mask_time_in_hours, ds, False, summary_writer)


def train_bm_model(model, model_path, summary_writer):
    ds = prepare_bm_dataset()
    train_model(model, model_path, bm_epochs, bm_time_in_hours, ds, True, summary_writer)


if __name__ == "__main__":
    model = load_model("model unet transformer.pth")
    model.to(device)
    summary(model.pre_model, input_size=(1, 3, 128, 128), device=device)