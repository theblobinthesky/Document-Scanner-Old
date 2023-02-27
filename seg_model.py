import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# UNet Transformer based on:
# https://arxiv.org/pdf/2103.06104.pdf

class Conv(nn.Module):
    def __init__(self, inp, out, kernel_size=3, padding=1, dilation=1):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=kernel_size, padding=padding, dilation=dilation, bias=False),
            nn.ReLU(True),
            nn.BatchNorm2d(out)
        )

    def forward(self, x):
        return self.layers(x)


class DoubleConv(nn.Module):
    def __init__(self, inp, out, mid=None):
        super().__init__()

        if mid == None:
            mid = out
        
        self.layers = nn.Sequential(
            nn.Conv2d(inp, mid, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(mid),
            nn.Conv2d(mid, out, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(out)
        )

    def forward(self, x):
        return self.layers(x)


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


def conv1x1(inp, out):
    return nn.Sequential(
        nn.Conv2d(inp, out, kernel_size=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(out),
    )

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
            self.bottle = Conv(depth[2], depth[2])
        
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
    

class DilatedBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.conv0 = Conv(inp, out // 2, kernel_size=3, padding=1)
        self.conv1 = Conv(out // 2, out // 4, kernel_size=3, padding=2, dilation=2)
        self.conv2 = Conv(out // 4, out // 8, kernel_size=3, padding=2, dilation=2)
        self.conv3 = Conv(out // 8, out // 16, kernel_size=3, padding=2, dilation=2)
        self.conv4 = Conv(out // 16, out // 16, kernel_size=3, padding=2, dilation=2)
        
        self.skip = conv1x1(inp, out)


    def forward(self, x):
        c0 = self.conv0(x)
        c1 = self.conv1(c0)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        skip = self.skip(x)

        return torch.cat([c0, c1, c2, c3, c4], axis=1) + skip


# Dilated Conv UNet based on:
# https://arxiv.org/pdf/2004.03466.pdf

class DownDilatedConv(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DilatedBlock(inp, out)
        )

    def forward(self, x):
        return self.layers(x)


class UpDilatedConv(nn.Module):
    def __init__(self, channelsY, channelsS):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2)
        self.block = DilatedBlock(channelsY + channelsS, channelsS)

    def forward(self, Y, S):
        Y = self.upsample(Y)
        I = torch.cat([Y, S], axis=1)
        O = self.block(I)

        return O


class OneArg(nn.Module):
    def __init__(self, layer, out):
        super().__init__()

        self.layer = layer
        self.conv = Conv(out, out)
    
    def forward(self, x):
        y = self.layer(x)
        y = self.conv(y)

        return y
    

class TwoArgs(nn.Module):
    def __init__(self, layer, out):
        super().__init__()

        self.layer = layer
        self.conv = Conv(out, out)
    
    def forward(self, y1, y2):
        y = self.layer(y1, y2)
        y = self.conv(y)

        return y


class UNetDilatedConv(nn.Module):
    def __init__(self, inp, out, large, think):
        super().__init__()

        depth = [32, 64, 128, 256]

        self.cvt_in = DoubleConv(inp, depth[0])
        self.cvt_out = DoubleConv(depth[0], out)

        if think:
            self.down0 = OneArg(DownDilatedConv(depth[0], depth[1]), depth[1])
            self.down1 = OneArg(DownDilatedConv(depth[1], depth[2]), depth[2])
        else:
            self.down0 = DownDilatedConv(depth[0], depth[1])
            self.down1 = DownDilatedConv(depth[1], depth[2])
        
        if large:
            self.down2 = DownDilatedConv(depth[2], depth[3])
            self.bottle = DilatedBlock(depth[3], depth[3])
            self.up2 = UpDilatedConv(depth[3], depth[2])
        else:
            self.bottle = DilatedBlock(depth[2], depth[2])

        self.large = large

        if think:
            self.up1 = TwoArgs(UpDilatedConv(depth[2], depth[1]), depth[1])
            self.up0 = TwoArgs(UpDilatedConv(depth[1], depth[0]), depth[0])
        else:
            self.up1 = UpDilatedConv(depth[2], depth[1])
            self.up0 = UpDilatedConv(depth[1], depth[0])
    

    def forward(self, x):
        x = self.cvt_in(x)

        if self.large:
            d0 = self.down0(x)
            d1 = self.down1(d0)
            d2 = self.down2(d1)

            bn = self.bottle(d2)

            u1 = self.up2(bn, d1)
            u0 = self.up1(u1, d0)
            y = self.up0(u0, x)
        else:
            d0 = self.down0(x)
            d1 = self.down1(d0)

            bn = self.bottle(d1)

            u0 = self.up1(bn, d0)
            y = self.up0(u0, x)

        y = self.cvt_out(y)
        return y