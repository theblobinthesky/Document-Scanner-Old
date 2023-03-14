import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF
from tqdm import tqdm

l2_weight = 0.3
grad_weight = 0.3
binarize_threshold = 0.5

class Conv(nn.Module):
    def __init__(self, inp, out, kernel_size=3, padding=1, dilation=1, stride=1, activation="none"):
        super().__init__()

        if activation == "relu":
            activation = nn.ReLU(True)
        elif activation == "sigmoid":
            activation = nn.Sigmoid()
        elif activation == "tanh":
            activation = nn.Tanh()
        elif activation == "none":
            activation = nn.Identity()
        else:
            print("error invalid activation")
            exit()

        self.layers = nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=kernel_size, padding=padding, dilation=dilation, stride=stride, bias=False),
            activation,
            nn.BatchNorm2d(out)
        )
    def forward(self, x):
        return self.layers(x)


def conv1x1(inp, out, padding=0, dilation=1, stride=1, activation="none"):
    return Conv(inp, out, 1, padding, dilation, stride, activation)


class DoubleConv(nn.Module):
    def __init__(self, inp, out, mid=None):
        super().__init__()

        if mid == None:
            mid = out
        
        self.layers = nn.Sequential(
            nn.Conv2d(inp, mid, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, out, kernel_size=3, padding=1, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(out)
        )

    def forward(self, x):
        return self.layers(x)


class DilatedBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.conv0 = DoubleConv(inp, out // 2)
        self.conv1 = Conv(out // 2, out // 4, kernel_size=3, padding=1, dilation=1, activation="relu")
        self.conv2 = Conv(out // 4, out // 8, kernel_size=3, padding=1, dilation=1, activation="relu")
        self.conv3 = Conv(out // 8, out // 16, kernel_size=3, padding=1, dilation=1, activation="relu")
        self.conv4 = Conv(out // 16, out // 16, kernel_size=3, padding=1, dilation=1, activation="relu")
        
        self.skip = conv1x1(inp, out)


    def forward(self, x):
        c0 = self.conv0(x)
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        skip = self.skip(x)

        return torch.cat([c0, c1, c2, c3, c4], axis=1) + skip


def metric_dice_coefficient_f(pred, label):
    pred, label = pred.detach(), label.detach()
    
    # todo: account for resizing. the label isnt quite right now
    return MF.dice(pred, label.int(), threshold=binarize_threshold).item()


def metric_specificity_f(pred, label):
    pred, label = pred.detach(), label.detach()
    
    # todo: account for resizing. the label isnt quite right now
    return MF.specificity(pred, label.int(), task="binary", threshold=binarize_threshold).item()


def metric_sensitivity_f(pred, label):
    pred, label = pred.detach(), label.detach()
    
    thresh = (pred > binarize_threshold).float()
    tp = (thresh * label).sum()
    fp = (thresh * (1.0 - label)).sum()

    return tp / (tp + fp).item()


def metric_local_distortion_f(pred, label):
    pred, label = pred.detach(), label.detach()

    return (pred - label).abs().mean()


def metric_line_distortion_f(pred, label):
    pred, label = pred.detach(), label.detach()

    local_distortion = (pred - label).abs()
    dx, dy = local_distortion[:, 0, :, :], local_distortion[:, 1, :, :]

    stdx = torch.std(dx, dim=2, unbiased=False)
    stdy = torch.std(dy, dim=1, unbiased=False)

    return stdx.mean() + stdy.mean()


metric_dice_coefficient = ("dice", metric_dice_coefficient_f)
metric_sensitivity = ("sensitivity", metric_specificity_f)
metric_specificity = ("specificity", metric_specificity_f)
metric_local_distortion = ("local_distortion", metric_local_distortion_f)
metric_line_distortion = ("line_distortion", metric_line_distortion_f)


def eval_loss_on_batches(model, iter, batch_count, device):
    loss = 0.0

    with torch.no_grad():
        for _ in range(batch_count):
            dict = next(iter)

            for key in dict.keys():
                dict[key] = dict[key].to(device)

            x, _ = model.input_and_label_from_dict(dict)
            pred = model(x)
    
            loss += model.loss(pred, dict).item()

    loss /= float(batch_count)

    return loss


def eval_loss_and_metrics_on_batches(model, iter, batch_count, device):
    loss = 0.0
    count = 0
    eval_metrics = model.eval_metrics()
    metrics = [0.0 for _ in eval_metrics]

    with torch.no_grad():
        for dict in tqdm(iter, desc="Evaluating test loss and metrics"):
            for key in dict.keys():
                dict[key] = dict[key].to(device)

            x, y = model.input_and_label_from_dict(dict)
            pred = model(x)
    
            loss += model.loss(pred, dict).item()
            count += 1

            for i, (_, func) in enumerate(eval_metrics):
                metrics[i] += func(pred, y)

    loss /= float(count)
    metrics = [(name, metrics[i] / float(count)) for i, (name, _) in enumerate(eval_metrics)]

    return loss, metrics


def loss_smooth(pred, label):
    diff = pred - label
    l1_loss = diff.abs().mean()
    l2_loss = diff.pow(2).mean()
    
    (pgrad_x, pgrad_y) = MF.image_gradients(pred)
    (lgrad_x, lgrad_y) = MF.image_gradients(label)
    grad_loss = 0.5 * (pgrad_x - lgrad_x).abs().mean() + 0.5 * (pgrad_y - lgrad_y).abs().mean()
    return l1_loss + l2_weight * l2_loss + grad_weight * grad_loss


def loss_circle_consistency(bm_pred, dict):
    bm_pred = 2.0 * bm_pred - 1.0
    bm_pred = bm_pred.permute(0, 2, 3, 1)

    uv_label = dict["uv"][:, :2]

    id_pred = F.grid_sample(uv_label, bm_pred, align_corners=False)

    loss_rows = id_pred - id_pred.mean(dim=3, keepdim=True)
    loss_cols = id_pred - id_pred.mean(dim=2, keepdim=True)
    
    loss_rows = loss_rows.sum()
    loss_cols = loss_cols.sum()

    return loss_rows + loss_cols


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model(model, path):
    torch.save(model.state_dict(), path)