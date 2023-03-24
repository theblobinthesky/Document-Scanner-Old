import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import device, l2_weight
from model import Conv, conv1x1, DoubleConv, MultiscaleBlock
from model import binarize_threshold, metric_l1, metric_dice_coefficient, metric_sensitivity, metric_specificity

min_finetuning_weight = torch.tensor(0.0, device=device)
max_finetuning_weight = torch.tensor(2.0, device=device)
finetuning_weight_amplitude = max_finetuning_weight - min_finetuning_weight
points_per_side_incl_start_corner = 4
contour_size = 4 * points_per_side_incl_start_corner * 3
feature_depth = 4 + 4 * points_per_side_incl_start_corner * 3 + 1

# UNet Transformer based on:
# https://arxiv.org/pdf/2103.06104.pdf

class Down(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(inp, out)
        )

    def forward(self, x):
        return self.layers(x)


class MultiHeadDense(nn.Module):
    def __init__(self, d):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(d, d))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        b, wh, d = x.size()
        return x @ self.weight.repeat(b, 1, 1)


def positional_encode(ten):
    _, d_model, height, width = ten.size()
    
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    try:
        pe = pe.to(device)
    except RuntimeError:
        pass
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(
        torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(
        0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(
        0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(
        0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(
        0, 1).unsqueeze(2).repeat(1, 1, width)
    
    return ten + pe

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()

        self.c = channels
        self.query = MultiHeadDense(channels)
        self.key = MultiHeadDense(channels)
        self.value = MultiHeadDense(channels)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.size()
        assert self.c == c

        x = positional_encode(x)

        x = x.reshape(b, c, h * w).permute(0, 2, 1)

        Q = self.query(x)
        K = self.key(x)
        
        A = self.softmax(Q @ K.permute(0, 2, 1) / math.sqrt(c))
        V = self.value(x)
        x = (A @ V).permute(0, 2, 1).reshape(b, c, h, w)
        return x
    

def convUp(channels):
    return nn.Sequential(
        nn.Upsample(scale_factor=2),
        nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    )

def convMask(channels):
    return nn.Sequential(
        nn.Conv2d(channels, channels, kernel_size=1),
        nn.Softmax(dim=1),
        nn.BatchNorm2d(channels),
        nn.Upsample(scale_factor=2)
    )

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, channelsY, channelsS):
        super().__init__()

        self.query = MultiHeadDense(channelsS)
        self.key = MultiHeadDense(channelsS)
        self.value = MultiHeadDense(channelsS)
        self.softmax = nn.Softmax(dim=1)

        self.Y1x1 = conv1x1(channelsY, channelsS)
        self.YLayers = nn.Sequential(
            convUp(channelsY),
            conv1x1(channelsY, channelsS)
        )
        
        self.SLayers = nn.Sequential(
            nn.MaxPool2d(2),
            conv1x1(channelsS, channelsS)
        )
        
        self.ZMask = convMask(channelsS)


    def forward(self, Y, S):
        Yb, Yc, Yh, Yw = Y.size()
        Sb, Sc, Sh, Sw = S.size()

        Y = positional_encode(Y)
        S = positional_encode(S)

        # critical shapes are annotated to keep track of the channel order
        Y1x1 = self.Y1x1(Y)
        Y1x1 = Y1x1.reshape((Yb, -1, Yh * Yw)) # shape = (b, c, h * w)
        Y1x1 = Y1x1.permute((0, 2, 1)) # shape = (b, h * w, c)

        Q = self.query(Y1x1) # shape = (b, h * w, c)
        K = self.key(Y1x1) # shape = (b, h * w, c)
        A = self.softmax(Q @ K.permute(0, 2, 1) / math.sqrt(Sc)) # shape = (b, h * w, h * w)

        S1x1 = self.SLayers(S).reshape((Sb, Sc, -1)) # shape = (b, c, h * w)
        S1x1 = S1x1.permute((0, 2, 1)) # shape = (b, h * w, c)
        V = self.value(S1x1) # shape = (b, h * w, c)

        Z = A @ V # shape = (b, h * w, c)
        Z = Z.permute((0, 2, 1)) # shape = (b, c, h * w)
        Z = Z.reshape((Yb, -1, Yh, Yw))
        Z = self.ZMask(Z) # shape = (b, c, h, w)

        O = S * Z
        O = torch.cat([self.YLayers(Y), O], axis=1)
        return O


class Up(nn.Module):
    def __init__(self, channelsY, channelsS):
        super().__init__()

        self.mhca = MultiHeadCrossAttention(channelsY, channelsS)
        self.YLayers = nn.Sequential(
            convUp(channelsY),
            conv1x1(channelsY, channelsY),
        )

        self.OConv = DoubleConv(2 * channelsY, channelsS)

    def forward(self, Y, S):
        mhca = self.mhca(Y, S)
        Y = self.YLayers(Y)
        
        O = torch.cat([mhca, Y], axis=1)
        O = self.OConv(O)
        return O


class UNetTransformer(nn.Module):
    def __init__(self, inp, out, use_self_attention=False):
        super().__init__()

        depth = [32, 64, 128, 256]

        self.cvt_in = DoubleConv(inp, depth[0])
        self.cvt_out = DoubleConv(depth[0], out)

        self.down0 = Down(depth[0], depth[1])
        self.down1 = Down(depth[1], depth[2])
        # self.down2 = Down(depth[2], depth[3])
        
        if use_self_attention:
            self.bottle = MultiHeadSelfAttention(depth[2])
        else:
            self.bottle = Conv(depth[2], depth[2], activation="relu")
        
        # self.up2 = Up(depth[3], depth[2])
        self.up1 = Up(depth[2], depth[1])
        self.up0 = Up(depth[1], depth[0])

    def forward(self, x):
        x = self.cvt_in(x)

        d0 = self.down0(x)
        d1 = self.down1(d0)
        # d2 = self.down2(d1)
        
        # mhsa = self.mhsa(d2)
        u2 = self.bottle(d1)

        # u2 = self.up2(mhsa, d1)
        u1 = self.up1(u2, d0)
        u0 = self.up0(u1, x)
        
        logits = self.cvt_out(u0)
        return logits
    

    def loss(self, pred, label):
        loss = F.binary_cross_entropy(pred, label)

        return loss


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

        inp = 4
        out = 1
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
        y = torch.sigmoid(y)

        return y


    def set_train(self, booleanlol):
        pass


    def input_and_label_from_dict(self, dict):
        img = dict["img"]

        b, _, h, w = img.shape
        a_padding = torch.full((b, 1, h, w), 1.0, device=device)

        img = torch.cat([img, a_padding], axis=1)
        mask = dict["uv"][:, 0].unsqueeze(1)

        return img, mask


    def loss(self, pred, dict, weight_metrics):
        mask_pred, _ = pred
        _, mask_label = self.input_and_label_from_dict(dict)

        mask_loss = F.binary_cross_entropy(mask_pred, mask_label, reduction="none")
        mask_loss = mask_loss.view(mask_loss.shape[0], -1).mean(1)
        
        finetuning = min_finetuning_weight + weight_metrics["finetuning"] * finetuning_weight_amplitude
        mask_loss = (finetuning * mask_loss).mean()
    
        return mask_loss
        

    def eval_metrics(self):
        return [] # todo: reimplement metrics return [metric_dice_coefficient, metric_sensitivity, metric_specificity]
    

# The Contour Model implements a modified much smaller version of a YOLOv3 object detector.

class FlattenEnd(nn.Module):
    def __init__(self, inp, hidden_channels, out, size):
        super().__init__()
        
        self.flatten = nn.Sequential(
            DoubleConv(inp, hidden_channels),
            nn.Flatten(),
            nn.Linear(size[0] * size[1] * hidden_channels, out),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.flatten(x)

class ContourModel(nn.Module):
    def __init__(self):
        super().__init__()

        inp, hidden_channels = 4, 16
        depth = [32, 64, 128, 256]

        self.cvt_in = DoubleConv(inp, depth[0])

        self.down0 = MultiscaleDown(depth[0], depth[1])
        self.down1 = MultiscaleDown(depth[1], depth[2])
        self.down2 = MultiscaleDown(depth[2], depth[3])
        
        self.bottle = nn.Sequential(
            MultiscaleBlock(depth[3], depth[3]),
            MultiscaleBlock(depth[3], depth[3])
        )

        # todo: remove the (8, 8) hack to allow for more resolution independence
        # self.exists = FlattenEnd(depth[3], hidden_channels, 1, (8, 8))
        self.feature = conv1x1(depth[3], feature_depth, activation="none")
        # self.contour = FlattenEnd(depth[3], hidden_channels, contour_size, (8, 8))
        
        
    def forward(self, x):
        x = self.cvt_in(x)

        d0 = self.down0(x)
        d1 = self.down1(d0)
        d2 = self.down2(d1)

        bn = self.bottle(d2)

        # ex = self.exists(bn)
        ft = self.feature(bn)

        # sigmoid center pt
        ft[:, 0:2, :, :] = torch.sigmoid(ft[:, 0:2, :, :]) 

        # sigmoid size
        ft[:, 2:4, :, :] = torch.exp(ft[:, 2:4, :, :]) 

        # sigmoid contour pt confidence
        contour_confidence_idx = [i for i in range(6, feature_depth - 2, 3)]
        ft[:, contour_confidence_idx, :, :] = torch.sigmoid(ft[:, contour_confidence_idx, :, :])

        # sigmoid objectness
        ft[:, -1, :, :] = torch.sigmoid(ft[:, -1, :, :])

        return ft


    def set_train(self, booleanlol):
        pass


    def input_and_label_from_dict(self, dict):
        img = dict["img"]

        b, _, h, w = img.shape
        a_padding = torch.full((b, 1, h, w), 1.0, device=device)

        img = torch.cat([img, a_padding], axis=1)    
        
        feature = dict["contour_feature/feature_map"]    
        cell = dict["contour_feature/cell"]
        
        return img, (feature, cell)


    def unpack_feature(self, feature):
        center_rel_to_cell = feature[:, 0:2, :, :]
        size = feature[:, 2:4, :, :]
        contour = feature[:, 4:-1, :, :]
        objectness = feature[:, -1, :, :].unsqueeze(1)

        return [center_rel_to_cell, size, contour, objectness]


    def loss(self, feature_pred, dict, weight_metrics):
        _, (feature_label, cell_label) = self.input_and_label_from_dict(dict)

        [center_rel_to_cell_label, size_label, contour_label, objectness_label] = self.unpack_feature(feature_label)
        [center_rel_to_cell_pred, size_pred, contour_pred, objectness_pred] = self.unpack_feature(feature_pred)
        
        center_loss = (center_rel_to_cell_label - center_rel_to_cell_pred).abs()
        size_loss = (size_label - size_pred).abs()
        contour_loss = (contour_label - contour_pred).abs()
        objectness_loss = F.binary_cross_entropy(objectness_pred, objectness_label).mean()

        def get_at_cell(ten):
            batch_indices = torch.arange(ten.shape[0])
            return ten[batch_indices, :, cell_label[:, 0], cell_label[:, 1]]
        
        coord_loss = get_at_cell(center_loss + size_loss).mean()
        contour_loss = get_at_cell(contour_loss).mean()

        return coord_loss + contour_loss + objectness_loss
        

    def eval_metrics(self, pred, label):
        feature_pred = pred
        feature_label = label

        return {
            "l1": 0.0 # metric_l1(contour_pred, contour_label)
        }


def binarize(x):
    x = torch.gt(x, binarize_threshold).float()

    return x