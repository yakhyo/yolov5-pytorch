import math

import torch
import torch.nn as nn


# Pad to 'same'
def _pad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


# Standard convolution
class Conv(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, _pad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


# Residual bottleneck
class Bottleneck(nn.Module):

    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.conv2(self.conv1(x)) if self.add else self.conv2(self.conv1(x))


# CSP Bottleneck with 3 convolutions
class C3(nn.Module):

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c1, c_, 1, 1)
        self.conv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.conv3(torch.cat((self.m(self.conv1(x)), self.conv2(x)), dim=1))


# Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
class SPPF(nn.Module):

    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.conv1 = Conv(c1, c_, 1, 1)
        self.conv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.conv1(x)

        y1 = self.m(x)
        y2 = self.m(y1)
        return self.conv2(torch.cat([x, y1, y2, self.m(y2)], 1))


# Concatenating list of layers
class Concat(nn.Module):

    def __init__(self, d=1):
        super().__init__()
        self.d = d

    def __call__(self, x: list):
        return torch.cat(x, self.d)


# Backbone
class BACKBONE(nn.Module):

    def __init__(self, filters, depths):
        super().__init__()

        self.b0 = Conv(filters[0], filters[1], 6, 2, 2)  # 0-P1/2
        self.b1 = Conv(filters[1], filters[2], 3, 2)  # 1-P2/4
        self.b2 = C3(filters[2], filters[2], depths[0])
        self.b3 = Conv(filters[2], filters[3], 3, 2)  # 3-P3/8
        self.b4 = C3(filters[3], filters[3], depths[1])
        self.b5 = Conv(filters[3], filters[4], 3, 2)  # 5-P4/16
        self.b6 = C3(filters[4], filters[4], depths[2])
        self.b7 = Conv(filters[4], filters[5], 3, 2)  # 7-P5/32
        self.b8 = C3(filters[5], filters[5], depths[0])
        self.b9 = SPPF(filters[5], filters[5], 5)  # 9

    def forward(self, x):
        b0 = self.b0(x)  # 0-P1/2
        b1 = self.b1(b0)  # 1-P2/4
        b2 = self.b2(b1)
        b3 = self.b3(b2)  # 3-P3/8
        b4 = self.b4(b3)
        b5 = self.b5(b4)  # 5-P4/16
        b6 = self.b6(b5)
        b7 = self.b7(b6)  # 7-P5/32
        b8 = self.b8(b7)
        b9 = self.b9(b8)

        return b4, b6, b9


# Head
class HEAD(nn.Module):

    def __init__(self, filters, depths):
        super().__init__()

        self.h10 = Conv(filters[5], filters[4], 1, 1)
        self.h11 = nn.Upsample(None, scale_factor=2, mode='nearest')
        self.h12 = Concat()  # cat backbone P4
        self.h13 = C3(filters[5], filters[4], depths[0], shortcut=False)  # 13

        self.h14 = Conv(filters[4], filters[3], 1, 1)
        self.h15 = nn.Upsample(None, scale_factor=2, mode='nearest')
        self.h16 = Concat()  # cat backbone P3
        self.h17 = C3(filters[4], filters[3], depths[0], shortcut=False)  # 17 (P3/8-small)

        self.h18 = Conv(filters[3], filters[3], 3, 2)
        self.h19 = Concat()  # cat head P4
        self.h20 = C3(filters[4], filters[4], depths[0], shortcut=False)  # 20 (P4/16-medium)

        self.h21 = Conv(filters[4], filters[4], 3, 2)
        self.h22 = Concat()  # cat head P5
        self.h23 = C3(filters[5], filters[5], depths[0], shortcut=False)  # 23 (P5/32-large)

    def forward(self, x):
        p3, p4, p5 = x

        h10 = self.h10(p5)
        h11 = self.h11(h10)
        h12 = self.h12([h11, p4])  # cat backbone P4
        h13 = self.h13(h12)  # 13

        h14 = self.h14(h13)
        h15 = self.h15(h14)
        h16 = self.h16([h15, p3])  # cat backbone P3
        h17 = self.h17(h16)  # 17 (P3/8-small)

        h18 = self.h18(h17)
        h19 = self.h19([h18, h14])  # cat head P4
        h20 = self.h20(h19)  # 20 (P4/16-medium)

        h21 = self.h21(h20)
        h22 = self.h22([h21, h10])  # cat head P5
        h23 = self.h23(h22)  # 23 (P5/32-large)

        return h17, h20, h23


# Detection Head
class DETECT(nn.Module):
    stride = None  # strides computed during build
    onnx_dynamic = False  # ONNX export parameter

    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        self.anchor_grid = [torch.zeros(1)] * self.nl  # init anchor grid
        self.register_buffer('anchors', torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv

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


# Model
class YOLOv5(nn.Module):

    def __init__(self, nc, anchors, conf):
        super(YOLOv5, self).__init__()

        filters = [3, 64, 128, 256, 512, 1024]
        depths = [3, 6, 9]
        depth_multiple, width_multiple = conf

        depths = [max(round(n * depth_multiple), 1) for n in depths]
        filters = [3, *[self._make_divisible(c * width_multiple, 8) for c in filters[1:]]]

        self.backbone = BACKBONE(filters, depths)
        self.head = HEAD(filters, depths)
        self.detect = DETECT(nc=nc, anchors=anchors, ch=(filters[3], filters[4], filters[5]))

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
    def _make_divisible(x, divisor):
        # Returns nearest x divisible by divisor
        if isinstance(divisor, torch.Tensor):
            divisor = int(divisor.max())  # to int
        return math.ceil(x / divisor) * divisor

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
