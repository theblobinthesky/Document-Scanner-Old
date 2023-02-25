import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics.functional as MF 
from torchvision.ops.focal_loss import sigmoid_focal_loss
from seg_model import UNetDilatedConv
from tqdm import tqdm

lam = 0.3
binarize_threshold = 0.8

class Conv(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, stride=1, padding='same', bias=False),
            nn.ReLU(),
            nn.BatchNorm2d(out)
        )
    
    def forward(self, x):
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, filters):
        super().__init__()

        self.conv0 = Conv(filters, filters)
        self.conv1 = Conv(filters, filters)
        
        
    def forward(self, inp):
        x = self.conv0(inp)
        x = self.conv1(x)
        x = x + inp

        return x


class BlockStack(nn.Module):
    def __init__(self, filters, blocks):
        super().__init__()

        self.layers = nn.Sequential(
            *[Block(filters) for _ in range(blocks)]
        )


    def forward(self, x):
        return self.layers(x)


class DownscaleBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(inp, out),
            BlockStack(out, 8)
        )


    def forward(self, x):
        return self.layers(x)


class UpscaleBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.conv = Conv(inp, 4 * out)
        self.stack = BlockStack(out, 8)


    def forward(self, x):
        x = self.conv(x)
        x = F.pixel_shuffle(x, 2)
        x = self.stack(x)
        
        return x


class UNet(nn.Module):
    def __init__(self, inp, out, blocks):
        super().__init__()

        self.blocks = blocks
        self.last = blocks - 1
        depth = [ 32, 64, 128, 256, 512, 1024 ]

        self.cvt_in = Conv(inp, depth[0])
            
        self.downs = nn.ModuleList([DownscaleBlock(depth[i], depth[i + 1]) for i in range(self.last)])

        self.bottle = Conv(depth[self.last], depth[self.last])

        self.ups = nn.ModuleList([UpscaleBlock(2 * depth[i], depth[i - 1]) for i in range(1, blocks).__reversed__()])

        self.cvt_out = Conv(2 * depth[0], out)
         

    def forward(self, x):
        x = self.cvt_in(x)

        x_ds = []
        for down in self.downs:
            x_ds.append(x)
            x = down(x)

        lx = x
        x = self.bottle(x)
        x = torch.cat([lx, x], axis=1)

        for i, up in enumerate(self.ups):
            x = up(x)
            x = torch.cat([x_ds[self.last - 1 - i], x], axis=1)
        
        x = self.cvt_out(x)

        return x
    

    def loss(self, pred, label):
        loss = F.binary_cross_entropy(pred, label)

        return loss


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

metric_dice_coefficient = ("dice", metric_dice_coefficient_f)
metric_sensitivity = ("sensitivity", metric_specificity_f)
metric_specificity = ("specificity", metric_specificity_f)

class PreModel(nn.Module):
    def __init__(self, focal_loss):
        super().__init__()

        self.focal_loss = focal_loss
        self.unet = UNetDilatedConv(3, 2)


    def forward(self, x):
        x = self.unet(x)
        x = torch.sigmoid(x)

        return x
    

    def loss(self, pred, label):
        if self.focal_loss:
            return sigmoid_focal_loss(pred, label, reduction="mean")
        else:
            return F.binary_cross_entropy(pred, label)


    def x_and_y_from_dict(self, dict):
        x = dict["img"]
        y = dict["lines"][:, [0, 2]]
        return x, y
    

    def eval_metrics(self):
        return [metric_dice_coefficient, metric_sensitivity, metric_specificity]


def binarize(x):
    x = torch.gt(x, binarize_threshold).float()

    return x
    

class WCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = UNet(4, 3, blocks=3)
        self.dummy = nn.Parameter(torch.empty(0))


    def forward(self, x):
        x = self.unet(x)
        x = torch.sigmoid(x)

        return x


    def loss(self, pred, label):
        return loss_smooth(pred, label)


class BMModel(nn.Module):
    def __init__(self):
        super().__init__()

        # self.net = ResNet(3, 2, 3, 4)
        self.net = UNet(3, 2, blocks=3)
        self.dummy = nn.Parameter(torch.empty(0))


    def forward(self, x):
        x = self.net(x)
        x = torch.sigmoid(x)

        return x


    def loss(self, pred, label):
        return loss_smooth(pred, label)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.pre_model = PreModel()
        self.wc_model = WCModel()
        self.bm_model = BMModel()

        self.dummy = nn.Parameter(torch.empty(0))
    

    def forward(self, inp):
        pre = self.pre_model(inp)
        pre = torch.sigmoid(pre)
        mask = pre[:, 0, :, :].unsqueeze(axis=1)
        lines = pre[:, 1, :, :].unsqueeze(axis=1)

        x = inp * mask
        x = torch.cat([x, lines], axis=1)
        x = self.wc_model(x)
        
        x = self.bm_model(x)

        return x


    def loss(self, pred, label):
        return loss_smooth(pred, label)


def eval_loss_on_batches(model, iter, batch_count, device):
    loss = 0.0

    with torch.no_grad():
        for _ in range(batch_count):
            dict = next(iter)

            for key in dict.keys():
                dict[key] = dict[key].to(device)

            x, y = model.x_and_y_from_dict(dict)
            pred = model(x)
    
            loss += model.loss(pred, y).item()

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

            x, y = model.x_and_y_from_dict(dict)
            pred = model(x)
    
            loss += model.loss(pred, y).item()
            count += 1

            for i, (_, func) in enumerate(eval_metrics):
                metrics[i] += func(pred, y)

    loss /= float(count)
    metrics = [(name, metrics[i] / float(count)) for i, (name, _) in enumerate(eval_metrics)]

    return loss, metrics


def loss_smooth(pred, label):
    abs_loss = (pred - label).abs().mean()
        
    (pgrad_x, pgrad_y) = MF.image_gradients(pred)
    (lgrad_x, lgrad_y) = MF.image_gradients(label)
    grad_loss = 0.5 * (pgrad_x - lgrad_x).abs().mean() + 0.5 * (pgrad_y - lgrad_y).abs().mean()
    return abs_loss + lam * grad_loss


def load_model(path):
    model = Model()
    model.load_state_dict(torch.load(path))

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)