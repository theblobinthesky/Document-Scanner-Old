import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import Conv, conv1x1, DoubleConv, DilatedBlock

# Progressive dewarping inspired by:
# https://arxiv.org/pdf/2110.14968.pdf

h_chs = 128
iters = 5

# class Upscale2(nn.Module):
#     def __init__(self, inp, out, t1):
#         super().__init__()

#         self.conv = nn.Sequential(
#             Conv(inp, out, kernel_size=1, padding=0, activation="relu") if t1 else Conv(inp, out, activation="relu"),
#             Conv(out, out * 4, activation="relu"),
#         )

#     def forward(self, x):
#         x = self.conv(x)
#         x = torch.pixel_shuffle(x, 2)

#         return x


class ConvGru(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.r_conv = Conv(h_chs + inp, h_chs, activation="sigmoid")
        self.z_conv = Conv(h_chs + inp, h_chs, activation="sigmoid")
        self.h_conv = Conv(2 * h_chs + inp, h_chs, activation="tanh")
        self.o_conv = DoubleConv(h_chs, out)

    def forward(self, Lh, x):
        Lh_x = torch.cat([Lh, x], axis=1)

        R = self.r_conv(Lh_x)
        R = Lh * R

        Z = self.z_conv(Lh_x)
        
        x_R = torch.cat([Lh, x, R], axis=1)
        H = self.h_conv(x_R)
        H = (1.0 - Z) * Lh + Z * H

        O = self.o_conv(H)
        return H, O


class ResNetBlockConst(nn.Module):
    def __init__(self, channels, dilated=False):
        super().__init__()

        dilation = 2 if dilated else 1
        padding = 2 if dilated else 1

        self.conv = nn.Sequential(
            Conv(channels, channels, dilation=dilation, padding=padding, activation="relu"),
            Conv(channels, channels, dilation=dilation, padding=padding, activation="relu")
        )

    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x + residual

        return x


class ResNetBlockChange(nn.Module):
    def __init__(self, inp, out, downscale):
        super().__init__()

        stride = 2 if downscale else 1

        self.conv = nn.Sequential(
            Conv(inp, inp, activation="relu"),
            Conv(inp, out, stride=stride, activation="relu")
        )

        self.skip = conv1x1(inp, out)
        
    def forward(self, x):
        residual = x
        x = self.conv(x)
        x = x + self.skip(residual)

        return x


class ResNetEncoder(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.layers = nn.Sequential(
            DilatedBlock(3, 64),

            nn.MaxPool2d(2),

            DilatedBlock(64, 64),
            ResNetBlockConst(64),
            
            nn.MaxPool2d(2),

            ResNetBlockConst(64),
            ResNetBlockConst(64),

            ResNetBlockChange(64, 128, False),
            conv1x1(128, 256, activation="relu")
        )

    def forward(self, x):
        return self.layers(x)

    
class ProgressiveModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.is_train = False

        dc_enc_chs = 256
        enc_all_inp_chs, enc_all_out_chs = 128, 128
        bm_enc_chs = 48
        dc_sampled_chs = enc_all_inp_chs - bm_enc_chs

        self.dc_enc = ResNetEncoder(3, dc_enc_chs)
        
        self.bm_enc = DoubleConv(2, bm_enc_chs)
        self.dc_sampled = DoubleConv(dc_enc_chs, dc_sampled_chs)
        self.enc_all = Conv(enc_all_inp_chs, enc_all_out_chs)

        self.gru = ConvGru(2 + enc_all_out_chs + dc_enc_chs, 2)

        self.upscaler = nn.Sequential(
            nn.UpsamplingBilinear2d((128, 128))
        )


    def set_train(self, mode):
        self.is_train = mode


    def forward_all(self, x):
        dc_enc = self.dc_enc(x)

        b, _, h, w = x.size()

        bm = torch.cartesian_prod(
            torch.linspace(0.0, 1.0, h, device="cuda"), 
            torch.linspace(0.0, 1.0, w, device="cuda")
        ).reshape(1, h, w, 2).permute(0, 3, 1, 2).repeat(b, 1, 1, 1)

        Lhs = [torch.zeros((b, h_chs, 32, 32), device="cuda")]
        bms = [bm]

        for _ in range(iters):
            Lh, bm_large = Lhs[-1], bms[-1]
            bm = F.interpolate(bm_large, (32, 32), mode="bilinear")

            bm_enc = self.bm_enc(bm)

            bm_sample = bm.permute(0, 2, 3, 1)
            bm_sample = bm_sample * 2.0 - 1.0
            dc_sampled = F.grid_sample(dc_enc, bm_sample, align_corners=False)
            dc_sampled = self.dc_sampled(dc_sampled)

            enc_all = self.enc_all(torch.cat([bm_enc, dc_sampled], axis=1))
            enc_all = torch.cat([bm, enc_all, dc_enc], axis=1)
            
            H, d_bm = self.gru(Lh, enc_all)

            d_bm = self.upscaler(d_bm)
            bm_large = bm_large + d_bm
            bm_large = torch.sigmoid(bm_large)

            Lhs.append(H)
            bms.append(bm_large)

        return bms


    def forward(self, x):
        ys = self.forward_all(x)
        return ys[-1]