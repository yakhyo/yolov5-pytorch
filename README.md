## YOLOv5 models

`yolov5.py`:

```python
if __name__ == '__main__':
    yolov5n = YOLOv5(nc=_nc, anchors=_anchors, conf=_conf['n'])
    yolov5s = YOLOv5(nc=_nc, anchors=_anchors, conf=_conf['s'])
    yolov5m = YOLOv5(nc=_nc, anchors=_anchors, conf=_conf['m'])
    yolov5l = YOLOv5(nc=_nc, anchors=_anchors, conf=_conf['l'])
    yolov5x = YOLOv5(nc=_nc, anchors=_anchors, conf=_conf['x'])

    print("Num. params of YOLOv5n: {}M".format(round(sum(p.numel() for p in yolov5n.parameters() if p.requires_grad) / 1e6, 1)))
    print("Num. params of YOLOv5s: {}M".format(round(sum(p.numel() for p in yolov5s.parameters() if p.requires_grad) / 1e6, 1)))
    print("Num. params of YOLOv5m: {}M".format(round(sum(p.numel() for p in yolov5m.parameters() if p.requires_grad) / 1e6, 1)))
    print("Num. params of YOLOv5l: {}M".format(round(sum(p.numel() for p in yolov5l.parameters() if p.requires_grad) / 1e6, 1)))
    print("Num. params of YOLOv5x: {}M".format(round(sum(p.numel() for p in yolov5x.parameters() if p.requires_grad) / 1e6, 1)))
```
Output
```
    Num. params of YOLOv5n: 1.9M
    Num. params of YOLOv5s: 7.2M
    Num. params of YOLOv5m: 21.2M
    Num. params of YOLOv5l: 46.6M
    Num. params of YOLOv5x: 86.7M
```
