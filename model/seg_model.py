import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import device
from model import DoubleConv, MultiscaleBlock
from model import binarize_threshold, metric_l1
import itertools

min_finetuning_weight = torch.tensor(0.0, device=device)
max_finetuning_weight = torch.tensor(2.0, device=device)
finetuning_weight_amplitude = max_finetuning_weight - min_finetuning_weight
points_per_side_incl_start_corner = 4
contour_pts = 4 * points_per_side_incl_start_corner

pos_weight = 100

# Dilated Conv UNet based on:
# https://arxiv.org/pdf/2004.03466.pdf
# The actual dilation is removed to increase inference performance on mobile gpus.

class MultiscaleDown(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            MultiscaleBlock(inp, out)
        )

    def forward(self, x):
        return self.layers(x)


class MultiscaleUp(nn.Module):
    def __init__(self, channelsY, channelsS):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.block = MultiscaleBlock(channelsY + channelsS, channelsS)

    def forward(self, Y, S):
        Y = self.upsample(Y)
        I = torch.cat([Y, S], axis=1)
        O = self.block(I)

        return O

class SegModel(nn.Module):
    def __init__(self):
        super().__init__()

        inp = 3
        out = 1
        depth = [64, 128, 256, 512]

        self.cvt_in = DoubleConv(inp, depth[0])
        self.cvt_out = DoubleConv(depth[0], out)

        self.down0 = MultiscaleDown(depth[0], depth[1])
        self.down1 = MultiscaleDown(depth[1], depth[2])
        self.down2 = MultiscaleDown(depth[2], depth[3])
        
        self.bottle = MultiscaleBlock(depth[3], depth[3])

        self.up2 = MultiscaleUp(depth[3], depth[2])
        self.up1 = MultiscaleUp(depth[2], depth[1])
        self.up0 = MultiscaleUp(depth[1], depth[0])


    def forward(self, x):
        x = self.cvt_in(x)

        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)

        bn = self.bottle(d2)

        u1 = self.up2(bn, d1)
        u0 = self.up1(u1, d0)
        y = self.up0(u0, x)

        y = self.cvt_out(y)
        y = torch.sigmoid(y)

        return y


    def set_train(self, booleanlol):
        pass


    def input_and_label_from_dict(self, dict):
        img = dict["img"]
        mask = dict["mask"]

        return img, mask


    def loss(self, mask_pred, dict, weight_metrics):
        _, mask_label = self.input_and_label_from_dict(dict)

        mask_loss = F.binary_cross_entropy(mask_pred, mask_label, reduction="none")
        mask_loss = mask_loss.view(mask_loss.shape[0], -1).mean(1)
        
        finetuning = min_finetuning_weight + weight_metrics["finetuning"] * finetuning_weight_amplitude
        mask_loss = (finetuning * mask_loss).mean()
    
        return mask_loss
        

    def eval_metrics(self, pred, label):
        return {} # todo: reimplement metrics return [metric_dice_coefficient, metric_sensitivity, metric_specificity]

class ContourModel(nn.Module):
    def __init__(self):
        super().__init__()

        inp = 4
        out = contour_pts
        depth = [32, 64, 128, 128]

        self.cvt_in = DoubleConv(inp, depth[0])
        self.cvt_out = DoubleConv(depth[0], out)

        self.down0 = MultiscaleDown(depth[0], depth[1])
        self.down1 = MultiscaleDown(depth[1], depth[2])
        self.down2 = MultiscaleDown(depth[2], depth[3])
        
        self.bottle = MultiscaleBlock(depth[3], depth[3])

        self.up2 = MultiscaleUp(depth[3], depth[2])
        self.up1 = MultiscaleUp(depth[2], depth[1])
        self.up0 = MultiscaleUp(depth[1], depth[0])
     
    def forward(self, x):
        x = self.cvt_in(x)

        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)

        bn = self.bottle(d2)

        u1 = self.up2(bn, d1)
        u0 = self.up1(u1, d0)
        y = self.up0(u0, x)

        y = self.cvt_out(y)
        
        return y


    def set_train(self, booleanlol):
        pass

    def get_brange_and_crange(self, b, c):
        brange = torch.arange(0, b).view(-1, 1).repeat(1, c)
        crange = torch.arange(0, c).view(1, -1).repeat(b, 1)
        return brange, crange

    def input_and_label_from_dict(self, dict):
        img = dict["img"]
        b, _, h, w = img.shape

        a_padding = torch.full((b, 1, h, w), 1.0, device=device)

        img = torch.cat([img, a_padding], axis=1)

        contour = dict["contour"]
        b, c, _ = contour.shape


        # generate heatmap
        heatmap = torch.zeros((b, c, w, h), device=device)

        brange, crange = self.get_brange_and_crange(b, c)
        
        xs, ys = contour[:, :, 0], contour[:, :, 1]
        xs, ys = xs * w, ys * w
        q_xs, q_ys = torch.floor(xs), torch.floor(ys)

        # choose random point based on errors
        e_x, e_y = xs - q_xs, ys - q_ys
        r_x, r_y = torch.rand(e_x.shape, device=device), torch.rand(e_y.shape, device=device)

        q_xs, q_ys = q_xs.int(), q_ys.int()
        i_x, i_y = (e_x >= r_x).int(), (e_y >= r_y).int()
        xs, ys = q_xs + i_x, q_ys + i_y

        heatmap[brange, crange, xs, ys] = 1.0

        return img, heatmap

    def contour_from_heatmap(self, heatmap, is_in_logits):
        b, c, w, h = heatmap.shape

        if is_in_logits:
            heatmap = torch.sigmoid(heatmap)

        # This is implementing Heatmap Regression via Randomized Rounding: 
        # https://arxiv.org/pdf/2009.00225.pdf

        # convert maximum indices to points
        indices = torch.argmax(heatmap.reshape(b, c, -1), dim=2)
        indices = indices.unsqueeze(-1)
        indices = torch.cat([indices / w, indices % w], axis=-1).int()

        # normalize 3x3 values in region around maximum and convolve with points
        brange, crange = self.get_brange_and_crange(b, c)
        xs, ys = indices[:, :, 0], indices[:, :, 1]

        def keep_in_bounds(ten, min, max):
            return torch.clamp(ten, min=min, max=max)

        kernel_arange = torch.arange(-1, 2)
        values = torch.cat([
            heatmap[brange, crange, keep_in_bounds(xs + xx, 0, w - 1), keep_in_bounds(ys + yy, 0, h - 1)].unsqueeze(-1) 
                for xx, yy in itertools.product(kernel_arange, kernel_arange)
        ], dim=-1)

        indices = torch.cat([
            torch.cat([keep_in_bounds(xs + xx, 0, w - 1).unsqueeze(-1).unsqueeze(-1), 
                       keep_in_bounds(ys + yy, 0, h - 1).unsqueeze(-1).unsqueeze(-1)], dim=-1) 
                for xx, yy in itertools.product(kernel_arange, kernel_arange)
        ], dim=-2)

        values /= values.sum(-1).unsqueeze(-1)
        values = values.unsqueeze(-1)

        contour = (indices * values).sum(2)
        contour[:, :, 0] /= w
        contour[:, :, 1] /= h

        return contour

    def loss(self, heatmap_pred, dict, weight_metrics):
        _, heatmap_label = self.input_and_label_from_dict(dict)

        _, c, h, w = heatmap_pred.shape
        pos_weight_ten = torch.tensor(pos_weight, device=device).expand(c, h, w)
        loss = F.binary_cross_entropy_with_logits(heatmap_pred, heatmap_label, pos_weight=pos_weight_ten, reduction="none")
        

        finetuning = min_finetuning_weight + weight_metrics["finetuning"] * finetuning_weight_amplitude
        
        b, _, _, _ = heatmap_pred.shape
        loss = finetuning.reshape(b, 1, 1, 1) * loss

        return loss.mean()

    def eval_metrics(self, pred, label):
        heatmap_pred = self.contour_from_heatmap(pred, is_in_logits=True)
        heatmap_label = self.contour_from_heatmap(label, is_in_logits=False)

        return {
            "L1": metric_l1(heatmap_pred, heatmap_label)
        }

def binarize(x):
    x = torch.gt(x, binarize_threshold).float()

    return x