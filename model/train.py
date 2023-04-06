#!/usr/bin/python3
from model import save_model, eval_loss_and_metrics_on_batches, count_params, device, Model
from data import load_seg_dataset, load_contour_dataset, load_bm_dataset
import torch
import torch.optim.lr_scheduler as lr_scheduler
import datetime
from itertools import cycle
from tqdm import tqdm
from benchmark import benchmark_plt_model
import select
import sys

warmup_lr = 1e-5
lr = 1e-3
steps_per_epoch = 40
seg_batch_size = 32
contour_batch_size = 32
bm_batch_size = 12

T_0 = 15
T_mul = 2
warmup_epochs = 15

def epochs_from_iters(iters):
    # - 2 epochs just makes sure you catch the local minimum
    return warmup_epochs + T_0 * sum([T_mul ** i for i in range(iters)]) - 2

seg_epochs = epochs_from_iters(4)
contour_epochs = epochs_from_iters(5)
bm_epochs = epochs_from_iters(6)
min_learning_rate_before_early_termination = 1e-7
lr_plateau_patience = 3
valid_batch_count = 16
valid_eval_every = 4
lam = 0.85

seg_time_in_hours = 2.0
contour_time_in_hours = 2.0
bm_time_in_hours = 4.0

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
        

def model_to_device(model):
    return model.to(device)


class CosineAnnealingWarmRestartsWithWarmup(lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self, optimizer, T_0, T_mult, min_warmup_lr, warmup_epochs):
        self.min_warmup_lr = min_warmup_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_epoch = 0
        super().__init__(optimizer, T_0, T_mult)

    def get_lr(self):
        if self.warmup_epoch < self.warmup_epochs:
            return [self.min_warmup_lr + (base_lr - self.min_warmup_lr) * self.warmup_epoch / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            return super().get_lr()

    def step(self):
        if self.warmup_epoch < self.warmup_epochs:
            lrs = self.get_lr()
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = lrs[i]

            self.warmup_epoch += 1
        else:
            super().step()



def train_model(model, model_path, model_type, summary_writer):
    if model_type == Model.SEG:
        epochs = seg_epochs
        time_in_hours = seg_time_in_hours
        batch_size = seg_batch_size
        ds = load_seg_dataset(batch_size)
    elif model_type == Model.CONTOUR:
        epochs = contour_epochs
        time_in_hours = contour_time_in_hours
        batch_size = contour_batch_size
        ds = load_contour_dataset(batch_size)
    elif model_type == Model.BM:
        epochs = bm_epochs
        time_in_hours = bm_time_in_hours
        batch_size = bm_batch_size
        ds = load_bm_dataset(batch_size)


    optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingWarmRestartsWithWarmup(optim, T_0, T_mul, warmup_lr, warmup_epochs)

    start_time = datetime.datetime.now()

    train_ds, valid_ds, test_ds = ds
    trainds_iter = iter(cycle(train_ds))
    valid_iter = iter(cycle(valid_ds))
    test_iter = iter(test_ds)

    pbar = tqdm(total=epochs)
    pbar.set_description("Initializing training and running first epoch")
    
    for epoch in range(epochs):
        train_loss = 0.0

        for _ in range(steps_per_epoch):
            dict, weight_metrics = next(trainds_iter)

            def dict_to_device(dict):
                return { key: value.to(device) for key, value in dict.items() }
            
            dict, weight_metrics = dict_to_device(dict), dict_to_device(weight_metrics)
                
            x, _ = model.input_and_label_from_dict(dict)

            optim.zero_grad(set_to_none=True)

            if model_type == Model.SEG or model_type == Model.CONTOUR:
                pred = model(x)
                loss = model.loss(pred, dict, weight_metrics)
            elif model_type == Model.BM:
                preds = model.forward_all(x)
                
                loss = 0.0

                for i, pred in enumerate(preds):
                    fac = lam ** (len(preds) - 1 - i)
                    loss += fac * model.loss(pred, dict, weight_metrics)

                loss /= float(len(preds))
                
            loss.backward()

            optim.step()

            train_loss += loss.detach().cpu().item()

        train_loss /= steps_per_epoch

        if epoch % valid_eval_every == 0:
            valid_loss, metrics = eval_loss_and_metrics_on_batches(model, valid_iter, "valid", valid_batch_count)
            
            summary_writer.add_scalar("Loss/valid", valid_loss, epoch)
            
            for (name, value) in metrics:
                summary_writer.add_scalar(name, value, epoch)

        scheduler.step()

        now = datetime.datetime.now()
        hours_passed = (now - start_time).seconds / (60.0 * 60.0)
        learning_rate = optim.param_groups[-1]['lr']

        summary_writer.add_scalar("Loss/train", train_loss, epoch)
        summary_writer.add_scalar("Learning rate", learning_rate, epoch)

        # check for keyboard input
        if select.select([sys.stdin], [], [], 0)[0]:
            input_str = sys.stdin.readline().strip()

            if input_str == "b":
                pbar.set_description("Benchmarking model")
                
                plt = benchmark_plt_model(model, model_type)
                summary_writer.add_figure("benchmark", plt)
                summary_writer.flush()
            elif input_str == "q": break


        pbar.update(1)
        pbar.set_description(f"Epoch {epoch + 1}/{epochs} completed with {hours_passed:.4f}h passed. Loss/train: {train_loss:.4f}, Loss/valid: {valid_loss:.4f}, LR: {learning_rate:.6f}")

        if hours_passed > time_in_hours:
            pbar.close()
            print(f"Hour limit of {time_in_hours:.4f}h passed.")
            break


    save_model(model, model_path)

    pbar.close()

    print("Evaluating test loss and metrics...")

    model.set_train(False)

    test_loss, metrics = eval_loss_and_metrics_on_batches(model, test_iter, "test", None)

    hparams = {"parameter_count": count_params(model)}

    metric_dict = {"Loss/test": test_loss}
    for (name, value) in metrics:
        metric_dict[name] = value

    summary_writer.add_hparams(hparams, metric_dict)

    print("Benchmarking model...")

    plt = benchmark_plt_model(model, model_type)
    summary_writer.add_figure("benchmark", plt)


if __name__ == "__main__":
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from torch.utils.tensorboard import SummaryWriter
    import seg_model, bm_model

    # print()
    # print("Training segmentation model")

    # writer = SummaryWriter("runs/seg_model_new4")
    # model = seg_model.PreModel()
    # model = model_to_device(model)
    # train_seg_model(model, "models/seg_model.pth", summary_writer=writer)
    # writer.flush()

    if True:
        print("Training contour model")

        name = "heatmap_7"
        writer = SummaryWriter(f"runs/{name}")
        model = seg_model.ContourModel()
        model = model_to_device(model)
        train_model(model, f"models/{name}.pth", Model.CONTOUR, summary_writer=writer)
        writer.flush()

    if False:
        print()
        print("Training bm model")

        name = "bm_model_3"
        writer = SummaryWriter(f"runs/{name}")
        model = bm_model.BMModel(True, False)
        model = model_to_device(model)
        train_model(model, f"models/{name}.pth", Model.BM, summary_writer=writer)
        writer.flush()