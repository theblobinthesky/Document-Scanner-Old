import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import image_gradients

lam = 0.3
binarize_threshold = 0.8


class Conv(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(inp, out, kernel_size=3, stride=1, padding='same', bias=False),
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
    def __init__(self, inp, out):
        super().__init__()

        self.layers = nn.Sequential(
            Conv(inp, out),
            Block(out),
            Block(out),
            Block(out)
        )


    def forward(self, x):
        return self.layers(x)


class DownscaleBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()
        
        self.layers = nn.Sequential(
            nn.MaxPool2d(2),
            BlockStack(inp, out)
        )

    def forward(self, x):
        return self.layers(x)


class UpscaleBlock(nn.Module):
    def __init__(self, inp, out):
        super().__init__()

        self.shuffle = Conv(inp, 4 * out)
        self.conv = Block(out)

    def forward(self, x):
        x = self.shuffle(x)
        x = F.pixel_shuffle(x, 2)
        x = self.conv(x)

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


class MaskModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = UNet(3, 1, blocks=2)
        self.bce = torch.nn.BCELoss()


    def forward(self, x):
        x = self.unet(x)
        x = torch.sigmoid(x)

        return x


    def loss(self, pred, label):
        loss = self.bce(pred, label)

        return loss


def binarize(x):
    x = torch.gt(x, binarize_threshold).float()

    return x
    

class WCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = UNet(3, 3, blocks=3)
        self.dummy = nn.Parameter(torch.empty(0))


    def forward(self, x):
        x = self.unet(x)
        x = torch.sigmoid(x)

        return x


    def loss(self, pred, label):
        return loss_smooth(pred, label)


    def eval_loss(self, ds):
        training = self.training
        self.train(False)

        dsiter = iter(ds)
        device = self.dummy.device

        loss, count = 0.0, 0
            
        for (img, uv_label) in dsiter:
            img, uv_label = img.to(device), uv_label.to(device)
            uv_pred = self(img)
                
            loss += self.loss(uv_pred, uv_label)
            count += 1

        self.train(training)

        return loss / float(count)


class BMModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.unet = UNet(3, 2, blocks=3)
        self.dummy = nn.Parameter(torch.empty(0))


    def forward(self, x):
        x = self.unet(x)
        x = torch.sigmoid(x)

        return x


    def loss(self, pred, label):
        return loss_smooth(pred, label)


    def eval_loss(self, ds):
        training = self.training
        self.train(False)

        dsiter = iter(ds)
        device = self.dummy.device

        loss, count = 0.0, 0
            
        for (img, uv_label) in dsiter:
            img, uv_label = img.to(device), uv_label.to(device)
            uv_pred = self(img)
                
            loss += self.loss(uv_pred, uv_label)
            count += 1

        self.train(training)

        return loss / float(count)


class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.mask_model = MaskModel()
        self.wc_model = WCModel()
        self.bm_model = BMModel()

        self.dummy = nn.Parameter(torch.empty(0))
    

    def forward(self, inp):
        mask = self.mask_model(inp)
        mask = torch.sigmoid(mask)

        x = inp * mask
        x = self.wc_model(x)
        
        x = self.bm_model(x)

        return x


    def loss(self, pred, label):
        return loss_smooth(pred, label)
    


def loss_smooth(pred, label):
    abs_loss = (pred - label).abs().mean()
        
    (pgrad_x, pgrad_y) = image_gradients(pred)
    (lgrad_x, lgrad_y) = image_gradients(label)
    grad_loss = 0.5 * (pgrad_x - lgrad_x).abs().mean() + 0.5 * (pgrad_y - lgrad_y).abs().mean()
    return abs_loss + lam * grad_loss


def load_model(path):
    model = Model()
    model.load_state_dict(torch.load(path))

    return model


def save_model(model, path):
    torch.save(model.state_dict(), path)