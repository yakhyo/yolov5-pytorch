import torch
from common import YOLOv5

_nc = 80

_anchors = [
    [10, 13, 16, 30, 33, 23],  # P3/8
    [30, 61, 62, 45, 59, 119],  # P4/16
    [116, 90, 156, 198, 373, 326]  # P5/32
]

_conf = {'n': (0.33, 0.25),
         's': (0.33, 0.5),
         'm': (0.67, 0.75),
         'l': (1.0, 1.0),
         'x': (1.33, 1.25)}


def yolov5(nc, anchors, conf):
    return YOLOv5(nc, anchors, conf)


def out(net):
    net.eval()
    img = torch.randn(1, 3, 640, 640)
    prediction, (p3, p4, p5) = net(img)
    return f'\nP3: {p3.shape}, \nP4: {p4.shape}, \nP5: {p5.shape}'


if __name__ == '__main__':
    yolov5n = yolov5(_nc, _anchors, _conf['n'])
    yolov5s = yolov5(_nc, _anchors, _conf['s'])
    yolov5m = yolov5(_nc, _anchors, _conf['m'])
    yolov5l = yolov5(_nc, _anchors, _conf['l'])
    yolov5x = yolov5(_nc, _anchors, _conf['x'])

    print("YOLOv5n params.: {:0.2f}M".format(sum(p.numel() for p in yolov5n.parameters() if p.requires_grad) / 1e6))
    print("YOLOv5s params.: {:0.2f}M".format(sum(p.numel() for p in yolov5s.parameters() if p.requires_grad) / 1e6))
    print("YOLOv5m params.: {:0.2f}M".format(sum(p.numel() for p in yolov5m.parameters() if p.requires_grad) / 1e6))
    print("YOLOv5l params.: {:0.2f}M".format(sum(p.numel() for p in yolov5l.parameters() if p.requires_grad) / 1e6))
    print("YOLOv5x params.: {:0.2f}M".format(sum(p.numel() for p in yolov5x.parameters() if p.requires_grad) / 1e6))
