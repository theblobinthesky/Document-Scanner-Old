import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Conv, conv1x1, DoubleConv, MultiscaleBlock
from model import binarize_threshold, metric_dice_coefficient, metric_sensitivity, metric_specificity

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
        pe = pe.to(torch.device("cuda:0"))
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


class UNetDilatedConv(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

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
    
min_finetuning_weight = torch.tensor(1.0, device="cuda")
max_finetuning_weight = torch.tensor(1.5, device="cuda")
finetuning_weight_amplitude = max_finetuning_weight - min_finetuning_weight

class PreModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = UNetDilatedConv(4, 1)

    def forward(self, x):
        x = self.unet(x)
        x = torch.sigmoid(x)

        return x


    def set_train(self, booleanlol):
        pass


    def input_and_label_from_dict(self, dict):
        img = dict["img"]

        b, _, h, w = img.shape
        a_padding = torch.full((b, 1, h, w), 1.0, device="cuda")

        img = torch.cat([img, a_padding], axis=1)

        if "uv" in dict:
            mask = dict["uv"][:, 0].unsqueeze(1)
        else:
            mask = torch.zeros((b, 1, h, w), device="cuda")

        return img, mask


    def loss(self, pred, dict, weight_metrics):
        _, label = self.input_and_label_from_dict(dict)

        loss = F.binary_cross_entropy(pred, label, reduction="none")
        loss = loss.view(loss.shape[0], -1).mean(1)

        finetuning = min_finetuning_weight + weight_metrics["finetuning"] * finetuning_weight_amplitude
        loss = (finetuning * loss).mean()
        return loss


    def eval_metrics(self):
        return [metric_dice_coefficient, metric_sensitivity, metric_specificity]


def binarize(x):
    x = torch.gt(x, binarize_threshold).float()

    return x