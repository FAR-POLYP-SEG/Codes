# FAR-POLYP-SEG — Baseline Segmentation Benchmark Code

This repository contains the training and evaluation code for the baseline experiments reported in the **FAR-POLYP-SEG** dataset paper. It benchmarks six deep-learning segmentation architectures on the FAR-POLYP-SEG colonoscopy dataset and additionally evaluates their zero-shot generalization on the public **Kvasir-SEG** dataset (no retraining).

## 1. Dataset

**FAR-POLYP-SEG** is a prospectively collected colonoscopy dataset of 8,040 frames from 455 patients: 432 polyp-positive frames with pixel-level segmentation masks and 7,608 frames of normal colonic mucosa. The segmentation models in this repository are trained and validated on the 432 mask-bearing (polyp-positive) frames, using patient-level splits so that frames from the same patient never appear in more than one split.

| Role | Source | Images | Usage |
| :--- | :--- | :--- | :--- |
| Source domain (internal) | FAR-POLYP-SEG (polyp-positive frames) | 432 | Patient-level train / validation / test splits; metrics averaged over 5 random seeds |
| Target domain (external) | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | 1000 | Held-out test set only — never seen during training or validation |

## 2. Implemented Models

1. **UNet** (`resnet34` encoder) — standard medical-segmentation baseline.
2. **UNet++** (`resnet34` encoder) — nested U-Net.
3. **TransUNet** (`mit_b0` encoder) — hybrid CNN–Transformer.
4. **nnU-Net** (2D) — self-configuring "no-new-Net" framework.
5. **YOLOv11m-seg** — Ultralytics real-time instance-segmentation model.
6. **PraNet** — parallel reverse-attention network.

## 3. Repository Layout

```
no-external-Val/                       # Internal FAR-POLYP-SEG benchmark (5 random seeds, no external validation)
├── Unet_multi.ipynb
├── UnetPP_multi.ipynb
├── TransUnet_multi.ipynb
├── nnUnet2D_multi_with_Inference.ipynb
├── YOLOv11_multi.ipynb
└── PraNet_multi.ipynb

Unet_KVASIR/        Unet_KVASIR_SEG.py        # Train on FAR-POLYP-SEG, evaluate zero-shot on Kvasir-SEG
UnetPP_KVASIR/      UnetPP_KVASIR_SEG.py
TransUnet_KVASIR/   TransUnet_KVASIR_SEG.py
nnUnet2D_KVASIR/    nnUnet2D_KVASIR_SEG.py
YOLOv11_KVASIR/     YOLOv11_KVASIR_SEG.py
PraNet_KVASIR/      PraNet_KVASIR_SEG.py
```

## 4. Zero-shot Results on Kvasir-SEG

| Model | Dice | IoU | Precision | Recall | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| UNet (ResNet34) | 0.8067 | 0.7185 | **0.9370** | 0.7582 | 0.9419 |
| UNet++ (ResNet34) | 0.8234 | 0.7382 | 0.9291 | 0.7863 | 0.9462 |
| TransUNet (MiT-B0) | **0.8361** | **0.7507** | 0.9216 | **0.8096** | **0.9501** |
| nnU-Net (2D) | 0.8268 | 0.7477 | 0.9306 | 0.7955 | 0.9480 |
| YOLOv11m-seg | 0.7763 | 0.7040 | 0.9108 | 0.7679 | 0.9362 |
| PraNet | 0.8118 | 0.7175 | 0.9001 | 0.7847 | 0.9437 |

Internal FAR-POLYP-SEG test-set results (mean ± standard deviation over 5 seeds), per-image inference latency, and the full comparison are reported in the paper.

## 5. Reproducing

- **Internal benchmark** — run the notebook for each architecture in `no-external-Val/`.
- **External validation** — `cd <Model>_KVASIR && python <Model>_KVASIR_SEG.py`. Each script downloads the datasets, prepares the masks/polygons, trains on FAR-POLYP-SEG, and evaluates on Kvasir-SEG.
