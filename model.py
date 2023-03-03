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
    

    # def loss(self, pred, dict):
    #     loss = F.binary_cross_entropy(pred, label)

    #     return loss


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


def metric_local_distortion(pred, label):
    pred, label = pred.detach(), label.detach()

    return (pred - label).abs().mean()


def metric_line_distortion(pred, label):
    pred, label = pred.detach(), label.detach()

    local_distortion = (pred - label).abs()
    dx, dy = local_distortion[:, 0, :, :], local_distortion[:, 1, :, :]

    stdx = torch.std(dx, dim=2, unbiased=False)
    stdy = torch.std(dy, dim=1, unbiased=False)

    return stdx.mean() + stdy.mean()


metric_dice_coefficient = ("dice", metric_dice_coefficient_f)
metric_sensitivity = ("sensitivity", metric_specificity_f)
metric_specificity = ("specificity", metric_specificity_f)

class PreModel(nn.Module):
    def __init__(self, large=False):
        super().__init__()

        self.unet = UNetDilatedConv(3, 2, large)


    def forward(self, x):
        x = self.unet(x)
        x = torch.sigmoid(x)

        return x
    

    def loss(self, pred, label):
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


from bm_model import ProgressiveModel
from bm_model import iters

cc_loss_enabled = False
alpha = 0.5
lam = 0.85

class BMModel(nn.Module):
    def __init__(self, train, learn_up, t1):
        super().__init__()

        self.net = ProgressiveModel(learn_up, t1)
        self.net.set_train(train)

        self.dummy = nn.Parameter(torch.empty(0))


    def set_train(self, train):
        self.net.set_train(train)


    def forward(self, x):
        y = self.net(x)

        return y


    def forward_all(self, x):
        return self.net.forward_all(x)


    def loss(self, preds, dict):
        total_loss = 0.0

        for i, pred in enumerate(preds):
            fac = lam ** (iters - 1 - i)
   
            if cc_loss_enabled:
                label = dict["bm"][:, :2]
                loss = loss_smooth(pred, label) + alpha * loss_circle_consistency(pred, dict)
            else:
                label = dict["bm"][:, :2]
                loss = loss_smooth(pred, label)

            total_loss += fac * loss

        return total_loss / float(len(preds))



    def x_and_y_from_dict(self, dict):
        x = dict["img_masked"]
        y = dict["bm"][:, :2]
        return x, y
    

    def eval_metrics(self):
        return [("local_distortion", metric_local_distortion)]


def eval_loss_on_batches(model, iter, batch_count, device):
    loss = 0.0

    with torch.no_grad():
        for _ in range(batch_count):
            dict = next(iter)

            for key in dict.keys():
                dict[key] = dict[key].to(device)

            x, _ = model.x_and_y_from_dict(dict)
            pred = model(x)
    
            loss += model.loss([pred], dict).item()

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
    
            loss += model.loss([pred], dict).item()
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


def loss_circle_consistency(bm_pred, dict):
    b, _, h, w = bm_pred.size()

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


def load_model(path):
    model = Model()
    model.load_state_dict(torch.load(path))

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)