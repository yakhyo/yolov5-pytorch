import math

import torch
import torch.nn as nn

anchors = [
    [10, 13, 16, 30, 33, 23],  # P3/8
    [30, 61, 62, 45, 59, 119],  # P4/16
    [116, 90, 156, 198, 373, 326]]  # P5/32


# Pad to 'same'
def _pad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


# Standard convolution
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _pad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# Standard bottleneck
class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


# CSP Bottleneck with 3 convolutions
class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


# Spatial Pyramid Pooling (SPP) layer [https://arxiv.org/abs/1406.4729]
class SPP(nn.Module):
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.conv1(x)
        return self.conv2(torch.cat([x] + [m(x) for m in self.m], 1))


# Spatial Pyramid Pooling - Fast (SPPF)
class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class DarkNet(nn.Module):
    def __init__(self):
        super(DarkNet, self).__init__()
        self.b0 = Conv(3, 64, 6, 2, 2)  # 0-P1/2
        self.b1 = Conv(64, 128, 3, 2)  # 1-P2/4
        self.b2 = C3(128, 128, 3)  # 2
        self.b3 = Conv(128, 256, 3, 2)  # 3-P3/8
        self.b4 = C3(256, 256, 6)  # 4
        self.b5 = Conv(256, 512, 3, 2)  # 5-P4/16
        self.b6 = C3(512, 512, 9)  # 6
        self.b7 = Conv(512, 1024, 3, 2)  # 7-P5/32
        self.b8 = C3(1024, 1024, 3)  # 8
        self.b9 = SPPF(1024, 1024, 5)  # 9

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1(b0)
        b2 = self.b2(b1)
        b3 = self.b3(b2)
        b4 = self.b4(b3)
        b5 = self.b5(b4)
        b6 = self.b6(b5)
        b7 = self.b7(b6)
        b8 = self.b8(b7)
        b9 = self.b9(b8)

        return b4, b6, b9


class Head(nn.Module):
    def __init__(self):
        super(Head, self).__init__()
        self.h10 = Conv(1024, 512, 1, 1)  # 10
        self.h11 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 11
        # self.h12 cat backbone P4 # 12
        self.h13 = C3(1024, 512, 3, shortcut=False)  # 13

        self.h14 = Conv(512, 256, 1, 1)  # 14
        self.h15 = nn.Upsample(None, scale_factor=2, mode='nearest')  # 15
        # self.h16 cat backbone P3 # 16
        self.h17 = C3(512, 256, 3, shortcut=False)  # 17 (P3/8-small)

        self.h18 = Conv(256, 256, 3, 2)  # 18
        # self.19 cat head P4 # 19
        self.h20 = C3(512, 512, 3, shortcut=False)  # 20 (P4/16-medium)

        self.h21 = Conv(512, 512, 3, 2)  # 21
        # self.h22 cat head P5 # 22
        self.h23 = C3(1024, 10243, shortcut=False)  # 23 23 (P5/32-large)

    def forward(self, x):
        p3, p4, p5 = x

        h10 = self.h10(p5)
        h11 = self.h11(h10)
        h12 = torch.cat([h11, p4], dim=1)  # cat backbone P4
        h13 = self.h13(h12)  # 13

        h14 = self.h14(h13)
        h15 = self.h15(h14)
        h16 = torch.cat([h15, p3], dim=1)  # cat backbone P3
        h17 = self.h17(h16)

        h18 = self.h18(h17)
        h19 = torch.cat([h18, h14], dim=1)  # cat head P4
        h20 = self.h20(h19)  # 20 (P4/16-medium)

        h21 = self.h21(h20)
        h22 = torch.cat([h21, h10], dim=1)  # cat head P5
        h23 = self.h23(h22)  # 23 (P5/32-large)

        return h17, h20, h23


class Detect(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use in-place ops (e.g. slice assignment)

    def forward(self, x):
        z = []  # inference output
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.onnx_dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2 - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20, i=0):
        d = self.anchors[i].device

        yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)])
        grid = torch.stack((xv, yv), 2).expand((1, self.na, ny, nx, 2)).float()

        anchor_grid = (self.anchors[i].clone() * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(
            (1, self.na, ny, nx, 2)).float()
        return grid, anchor_grid


class YOLOv5(nn.Module):
    def __init__(self, anchors):
        super(YOLOv5, self).__init__()

        self.backbone = DarkNet()
        self.head = Head()
        self.detect = Detect(anchors=anchors, ch=(1024, 512, 256))

        dummy_img = torch.zeros(1, 3, 256, 256)
        self.detect.stride = torch.tensor([256 / x.shape[-2] for x in self.forward(dummy_img)])
        self.detect.anchors /= self.detect.stride.view(-1, 1, 1)
        self._check_anchor_order(self.detect)
        self._initialize_biases(self.detect)

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        p3, p4, p5 = self.head([p3, p4, p5])
        return self.detect([p3, p4, p5])

    @staticmethod
    def _initialize_biases(detect):
        det = detect
        for layer, stride in zip(det.m, det.stride):
            b = layer.bias.view(det.na, -1)
            b.data[:, 4] += math.log(8 / (640 / stride) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (det.nc - 0.999999))
            layer.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    @staticmethod
    def _check_anchor_order(det):
        a = det.anchors.prod(-1).view(-1)  # anchor area
        da = a[-1] - a[0]  # delta a
        ds = det.stride[-1] - det.stride[0]  # delta s
        if da.sign() != ds.sign():  # same order
            det.anchors[:] = det.anchors.flip(0)


if __name__ == '__main__':
    net = YOLOv5(anchors=anchors)
    # net.eval() # error occurs, fixing...
    img = torch.randn(1, 3, 640, 640)
    p3, p4, p5 = net(img)
    print(p3.shape, p4.shape, p5.shape)
    print("Num. of parameters: {}".format(sum(p.numel() for p in net.parameters() if p.requires_grad)))
