import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import image_gradients

lam = 0.2

class Conv(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, stride=1, padding='same', bias=False),
            nn.BatchNorm2d(out)
        )
    
    def forward(self, x):
        return self.layers(x)


class DoubleConv(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.layers = nn.Sequential(
            Conv(inp, out),            
            Conv(out, out),
        )

    def forward(self, x):
        return self.layers(x)


class DownscaleBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(inp, out)
        )

    def forward(self, x):
        return self.layers(x)


class UpscaleBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.conv0 = Conv(inp, 4 * inp)
        self.conv1 = Conv(inp, out)

    def forward(self, x):
        x = self.conv0(x)
        x = F.pixel_shuffle(x, 2)
        x = self.conv1(x)

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


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet0 = UNet(3, 3, blocks=4)

        #self.unet1 = UNet(5, 3, blocks=2)

        self.dummy = nn.Parameter(torch.empty(0))
    
    def forward(self, inp):
        x = self.unet0(inp)
        x = torch.sigmoid(x)
        
        #x = torch.cat([inp, x], axis=1)

        #x = self.unet1(x)
        #x = torch.sigmoid(x)

        return x
    
    def eval_loss_on_ds(self, ds):
        dsiter = iter(ds)
        device = self.dummy.device

        loss, count = 0.0, 0
        
        for (img, uv_label) in dsiter:
            img, uv_label = img.to(device), uv_label.to(device)
            uv_pred = self(img)
            
            loss += loss_function(uv_pred, uv_label)
            count += 1

        return loss / float(count)


def loss_function(uv_pred, uv_label):
    abs_loss = (uv_pred - uv_label).abs().mean()
    
    (pgrad_x, pgrad_y) = image_gradients(uv_pred)
    (lgrad_x, lgrad_y) = image_gradients(uv_label)
    grad_loss = 0.5 * (pgrad_x - lgrad_x).abs().mean() + 0.5 * (pgrad_y - lgrad_y).abs().mean()
    return abs_loss + lam * grad_loss


def load_model(path):
    model = Model()
    model.load_state_dict(torch.load(path))

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)