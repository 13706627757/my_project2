import argparse
import sys
from pathlib import Path

import cv2
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, increment_path, non_max_suppression, scale_boxes
from utils.torch_utils import select_device, smart_inference_mode


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv5 inference on a single image.")
    parser.add_argument(
        "--source",
        type=str,
        default=r"E:\study\jetson_nano\yolov5\yolov5\data\images\bus.jpg",
        help="path to the input image",
    )
    parser.add_argument("--weights", type=str, default=str(ROOT / "yolov5s.pt"), help="path to weights file")
    parser.add_argument("--imgsz", type=int, default=640, help="inference image size")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument("--device", default="", help="cuda device like 0 or cpu")
    parser.add_argument("--project", type=str, default=str(ROOT / "runs" / "image_test"), help="directory to save results")
    parser.add_argument("--name", type=str, default="exp", help="result subdirectory name")
    parser.add_argument("--exist-ok", action="store_true", help="reuse an existing result directory")
    return parser.parse_args()


@smart_inference_mode()
def main():
    args = parse_args()
    source = str(Path(args.source).expanduser())
    device = select_device(args.device)

    model = DetectMultiBackend(args.weights, device=device, dnn=False, data=None, fp16=False)
    stride, names = model.stride, model.names
    imgsz = check_img_size(args.imgsz, s=stride)

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=model.pt)
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    image_found = False
    for path, im, im0s, _, _ in dataset:
        image_found = True
        im = torch.from_numpy(im).to(model.device)
        im = im.float()
        im /= 255.0
        if im.ndim == 3:
            im = im.unsqueeze(0)

        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, max_det=1000)

        im0 = im0s.copy()
        det = pred[0]

        print(f"Image: {path}")
        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                x1, y1, x2, y2 = [int(v) for v in xyxy]
                label = f"{names[int(cls)]} {conf:.2f}"
                cv2.rectangle(im0, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(im0, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                print(f"  class={names[int(cls)]}, conf={conf:.2f}, box=({x1}, {y1}, {x2}, {y2})")
        else:
            print("  no detections")

        output_path = save_dir / Path(path).name
        cv2.imwrite(str(output_path), im0)
        print(f"Saved result to: {output_path}")
        break

    if not image_found:
        raise FileNotFoundError(f"Could not load image: {source}")


if __name__ == "__main__":
    main()
