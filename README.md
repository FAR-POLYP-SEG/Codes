# Polyp Segmentation Model Benchmarking

## 1. Project Overview
This project benchmarks various deep learning architectures for polyp segmentation. The goal is to train models on a **FAR-POLYP-SEG** dataset and evaluate their **generalization capability** on a completely unseen external dataset (**KVASIR-SEG**).

## 2. Dataset Strategy
We enforce a strict separation of data to ensure fair and robust evaluation:

### Primary Dataset (Source Domain)
*   **Source**: FAR-POLYP-SEG dataset.
*   **Total Images**: ~432.
*   **Usage**:
    *   **Training Set (80%)**: Used for model weight optimization.
    *   **Validation Set (20%)**: Used for hyperparameter tuning and early stopping.

### External Test Dataset (Target Domain)
*   **Source**: [KVASIR-SEG](https://datasets.simula.no/kvasir-seg/).
*   **Total Images**: 1000.
*   **Usage**: **Strictly Test**. This dataset is never seen during training or validation. It is used exclusively for the final performance metrics reported below.

## 3. Implemented Models
We have implemented and evaluated the following architectures:

1.  **YOLOv11** (`yolo11m-seg`): Ultralytics' latest real-time instance segmentation model.
2.  **PraNet**: A parallel reverse attention network designed for camouflaged object detection.
3.  **TransUnet**: A hybrid Transfomer-CNN architecture (`mit_b0` encoder).
4.  **Unet**: The standard medical segmentation baseline (`resnet34` encoder).
5.  **Unet++**: An improved nested U-Net architecture (`resnet34` encoder).
6.  **nnU-Net**: The self-configuring "no-new-Net" framework (2D configuration, currently training).

## 4. Benchmark Results (KVASIR-SEG Test Set)

| Model Name | Dice Coefficient | IoU (Jaccard) | Precision | Recall | Accuracy | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **YOLOv11** | 0.7763 | 0.7040 | 0.9108 | 0.7679 | 0.9362 | Fast inference, polygon output |
| **PraNet** | 0.8118 | 0.7175 | 0.9001 | 0.7847 | 0.9437 | Reverse attention mechanism |
| **TransUnet** | **0.8361** | **0.7507** | 0.9216 | **0.8096** | **0.9501** | **Current Best Performer** |
| **Unet (ResNet34)** | 0.8067 | 0.7185 | **0.9370** | 0.7582 | 0.9419 | Standard baseline |
| **Unet++ (ResNet34)** | 0.8234 | 0.7382 | 0.9291 | 0.7863 | 0.9462 | Nested U-Net architecture |
| **nnU-Net (2D)** | 0.8268 | 0.7477 | 0.9306 | 0.7955 | 0.9480 | **Final (1000 Epochs)** |

## 5. How to Reproduce

Each model has its own dedicated directory with an automated training script (`*_KVASIR_SEG.py`) and a deployment script (`deploy_*.py`).

### General Steps:
1.  Navigate to the model's directory (e.g., `cd UnetPP_KVASIR`).
2.  Run the training script (locally or remotely). The script automatically:
    *   Downloads both datasets.
    *   Formats data (masks/polygons).
    *   Splits Primary data (80/20).
    *   Trains the model.
    *   Evaluates on KVASIR.

### Directory Structure
```
models/
├── YOLOv11_KVASIR/      # YOLOv11 Implementation
├── PraNet_KVASIR/       # PraNet Implementation
├── TransUnet_KVASIR/    # TransUnet Implementation
├── Unet_KVASIR/         # Unet Implementation
├── UnetPP_KVASIR/       # Unet++ Implementation
└── nnUnet2D_KVASIR/     # nnU-Net Implementation
```
