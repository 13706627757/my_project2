import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv5 webcam test script")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(ROOT / "yolov5s.pt"),
        help="path to model weights file (.pt)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="webcam source, use 0 for default camera",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="inference image size",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="confidence threshold",
    )
    parser.add_argument(
        "--iou-thres",
        type=float,
        default=0.45,
        help="NMS IoU threshold",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="",
        help="cuda device, i.e. 0 or cpu",
    )
    return parser.parse_args()


def plot_one_box(xyxy, img, color=(0, 255, 0), label=None, line_thickness=2):
    x1, y1, x2, y2 = [int(x) for x in xyxy]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=line_thickness, lineType=cv2.LINE_AA)
    if label:
        tf = max(line_thickness - 1, 1)
        text_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=tf)[0]
        c2 = x1 + text_size[0], y1 - text_size[1] - 3
        cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)
        cv2.putText(img, label, (x1, y1 - 2), 0, 0.5, (255, 255, 255), thickness=tf, lineType=cv2.LINE_AA)


def main():
    args = parse_args()

    device = torch.device("cpu")
    if args.device != "cpu" and torch.cuda.is_available():
        device = torch.device("cuda:" + str(args.device)) if args.device != "" else torch.device("cuda:0")

    source = args.source
    if source.isnumeric():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {args.source}")

    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=None, fp16=False)
    stride = int(model.stride)
    imgsz = check_img_size(args.imgsz, s=stride)
    names = model.names
    half = model.fp16

    model.warmup(imgsz=(1, 3, imgsz, imgsz))

    print(f"加载模型: {args.weights}")
    print(f"使用设备: {device}")
    print("按 q 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取摄像头帧，正在退出")
            break

        im0 = frame.copy()
        img = letterbox(im0, imgsz, stride=stride, auto=True)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)

        im = torch.from_numpy(img).to(device)
        im = im.half() if half else im.float()
        im /= 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, max_det=1000)

        for det in pred:
            if len(det):
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    label = f"{names[int(cls)]} {conf:.2f}"
                    plot_one_box(xyxy, im0, label=label)

        cv2.imshow("YOLOv5 Webcam", im0)
        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
