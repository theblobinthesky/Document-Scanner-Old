import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, stride=1, padding='same'),
            nn.Conv2d(out, out, kernel_size=3, stride=1, padding='same')
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

        self.conv0 = nn.Conv2d(inp, 4 * inp, kernel_size=3, stride=1, padding='same')
        self.conv1 = nn.Conv2d(inp, out, kernel_size=3, stride=1, padding='same')
            
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
        depth = [ 16, 32, 64, 128, 256, 512, 1024 ]

        self.cvt_in = nn.Conv2d(inp, depth[0], kernel_size=3, stride=1, padding='same')

        self.downs = nn.ModuleList([DownscaleBlock(depth[i], depth[i + 1]) for i in range(self.last)])

        self.bottle = nn.Conv2d(depth[self.last], depth[self.last], kernel_size=3, stride=1, padding='same')

        self.ups = nn.ModuleList([UpscaleBlock(2 * depth[i], depth[i - 1]) for i in range(1, blocks).__reversed__()])

        self.cvt_out = nn.Conv2d(2 * depth[0], out, kernel_size=3, stride=1, padding='same')

    def forward(self, x):
        x = self.cvt_in(x)
        
        x_ds = []
        for down in self.downs:
            x_ds.append(x)
            x = down(x)

        lx = x
        x = self.bottle(x)
        x = torch.concatenate([lx, x], axis=1)

        for i, up in enumerate(self.ups):
            x = up(x)
            x = torch.concatenate([x_ds[self.last - 1 - i], x], axis=1)
        
        x = self.cvt_out(x)

        return x


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet0 = UNet(3, 2, blocks=3)

        self.unet1 = UNet(2, 3, blocks=3)

        self.dummy = nn.Parameter(torch.empty(0))
    
    def forward(self, x):
        x = self.unet0(x)
        
        x = self.unet1(x)

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
    return (uv_pred - uv_label).abs().mean()

def load_model(path):
    model = Model()
    model.load_state_dict(torch.load(path))

    return model

def save_model(model, path):
    torch.save(model.state_dict(), path)