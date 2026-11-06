# Pimple Removal Tool

Upload a face photo to compare acne removal by DeepFillv2 and OpenCV Telea inpainting. A YOLOv8s detector trained on Acne04 finds the acne regions. Both methods use the same detected mask. The app shows each result and provides draggable sliders for original versus DeepFillv2 and OpenCV versus DeepFillv2.

## Run the demo

In PowerShell, from this repository:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Open the local address printed by Gradio (usually http://127.0.0.1:7860), upload a face image, and click **Remove Pimples**. Press Ctrl+C to stop the app.

The trained detector is already included at `models/acne04_60_best.pt`, so uploading a new image does not retrain it. The first launch downloads DeepFillv2's inference code and pretrained weights and needs internet access. Later launches reuse those files. OpenCV Telea is a classical algorithm and needs no trained checkpoint. Processing may take longer on a CPU.

To use a different YOLO checkpoint, set `ACNE_YOLO_MODEL_PATH` before launching. To request a public Gradio share link, set `GRADIO_SHARE=1`.

## Training

The included YOLOv8s detector was trained for 60 epochs on the [Acne04 YOLOv8 dataset](https://www.kaggle.com/datasets/shreyash1110/acne04-yolov8). Its best validation mAP@50 was approximately 0.325 at epoch 57. The dataset and full training outputs are excluded from GitHub; only the best checkpoint is committed.

To reproduce the run, download that dataset and arrange it as:

```text
data/acne04-yolov8/
  Images/*.jpg
  labels/content/labels/*.txt
```

Then run:

```powershell
.\.venv\Scripts\python.exe train_acne04.py --epochs 60 --batch 16
```

The script creates a 70/20/10 train/validation/test split and saves outputs in `runs/yolov8s_acne_augmented_60/`. A smaller GPU may need `--batch 4`. To continue an interrupted run, use `--resume`. After retraining, copy its `weights/best.pt` to `models/acne04_60_best.pt` to use it in the app.

The comparison sliders show visual differences. Acne04 provides lesion boxes rather than paired clean target images, so these outputs cannot establish a numeric skin restoration accuracy score.

## Source credit

This app's YOLOv8s and DeepFillv2 approach was adapted from [architk22/EE604-Course-Project](https://github.com/architk22/EE604-Course-Project). DeepFillv2 inference code and weights come from [vrindaprabhu/deepfillv2_colab](https://github.com/vrindaprabhu/deepfillv2_colab). The OpenCV baseline, comparison interface, training script, and 60-epoch detector checkpoint are included in this repository.
