"""Train a YOLOv8s acne detector on Acne04 for 60 epochs by default.

Expected raw dataset layout:
  data/acne04-yolov8/Images/*.jpg
  data/acne04-yolov8/labels/content/labels/*.txt
"""

import argparse
import os
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split
from ultralytics import YOLO
import torch


ROOT = Path(__file__).resolve().parent
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def prepare_dataset(source: Path, output: Path) -> Path:
    image_dir = source / "Images"
    label_dir = source / "labels" / "content" / "labels"
    if not image_dir.is_dir() or not label_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {image_dir} and {label_dir}. "
            "Place the Acne04 YOLOv8 download there or pass --source."
        )

    pairs = []
    for image in sorted(image_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label = label_dir / f"{image.stem}.txt"
        if label.is_file():
            pairs.append((image, label))
    if len(pairs) < 10:
        raise ValueError(f"Only {len(pairs)} image/label pairs found in {source}")

    train_val, test = train_test_split(pairs, test_size=0.1, random_state=42)
    train, val = train_test_split(train_val, test_size=2 / 9, random_state=42)
    for name, split in (("train", train), ("val", val), ("test", test)):
        images_out = output / "images" / name
        labels_out = output / "labels" / name
        images_out.mkdir(parents=True, exist_ok=True)
        labels_out.mkdir(parents=True, exist_ok=True)
        for image, label in split:
            shutil.copy2(image, images_out / image.name)
            shutil.copy2(label, labels_out / label.name)

    yaml_path = output / "data.yaml"
    yaml_path.write_text(
        f"path: {output.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "names:\n  0: pimple\n",
        encoding="utf-8",
    )
    print(f"Prepared Acne04: {len(train)} train, {len(val)} val, {len(test)} test")
    return yaml_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "acne04-yolov8")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "acne_dataset")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2 if os.name == "nt" else 8)
    parser.add_argument("--resume", action="store_true",
                        help="Resume the current 60 epoch run from last.pt")
    args = parser.parse_args()

    if args.resume:
        checkpoint = ROOT / "runs" / "yolov8s_acne_augmented_60" / "weights" / "last.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"No training checkpoint at {checkpoint}")
        YOLO(str(checkpoint)).train(resume=True)
        return

    data_yaml = prepare_dataset(args.source, args.output)
    device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO("yolov8s.pt")
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=640,
        batch=args.batch,
        name="yolov8s_acne_augmented_60",
        project=str(ROOT / "runs"),
        device=device,
        optimizer="AdamW",
        patience=args.epochs,
        workers=args.workers,
        exist_ok=True,
        verbose=True,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        shear=0.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        erasing=0.4,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
    )


if __name__ == "__main__":
    main()
