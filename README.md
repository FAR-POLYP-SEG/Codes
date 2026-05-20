# FAR-POLYP-SEG Baseline Segmentation Code

This repository contains the training and evaluation code for the baseline
experiments reported in the FAR-POLYP-SEG dataset paper. It benchmarks six
segmentation architectures on FAR-POLYP-SEG and evaluates the trained models on
Kvasir-SEG as an unseen external test dataset.

## Dataset Usage

FAR-POLYP-SEG contains 8,181 colonoscopy frames from 455 patients. The baseline
segmentation experiments use the 432 polyp-positive frames with pixel-level
segmentation masks. Patient-level splits are used for the internal benchmark so
that frames from the same patient do not appear in more than one split.

| Role | Dataset | Images | Usage |
| :--- | :--- | :--- | :--- |
| Internal benchmark | FAR-POLYP-SEG polyp-positive frames | 432 | Patient-level train, validation, and test splits; metrics averaged over five random seeds |
| External test | [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) | 1,000 | Unseen test dataset; not used for training, validation, model selection, or tuning |

## Models

1. UNet with a ResNet34 encoder
2. UNet++ with a ResNet34 encoder
3. TransUNet with a MiT-B0 encoder
4. nnU-Net 2D
5. YOLOv11m-seg
6. PraNet

## Repository Layout

```text
FAR-POLYP-SEG/
├── data.zip
├── code/
│   ├── internal/
│   │   ├── unet_internal_benchmark.ipynb
│   │   ├── unetpp_internal_benchmark.ipynb
│   │   ├── transunet_internal_benchmark.ipynb
│   │   ├── nnunet2d_internal_benchmark.ipynb
│   │   ├── yolov11_internal_benchmark.ipynb
│   │   └── pranet_internal_benchmark.ipynb
│   └── external/
│       ├── unet_kvasir_external_test.py
│       ├── unetpp_kvasir_external_test.py
│       ├── transunet_kvasir_external_test.py
│       ├── nnunet2d_kvasir_external_test.py
│       ├── yolov11_kvasir_external_test.py
│       └── pranet_kvasir_external_test.py
├── .gitignore
└── README.md
```

## Running the Internal Benchmark

Open the model-specific notebook in `code/internal/` and run all
cells. These notebooks train and evaluate models on FAR-POLYP-SEG using the
internal patient-level benchmark protocol.

## Running External Kvasir-SEG Tests

Run the model-specific script from the repository root:

```bash
python code/external/unet_kvasir_external_test.py
python code/external/unetpp_kvasir_external_test.py
python code/external/transunet_kvasir_external_test.py
python code/external/nnunet2d_kvasir_external_test.py
python code/external/yolov11_kvasir_external_test.py
python code/external/pranet_kvasir_external_test.py
```

Each script resolves paths from the repository root, downloads the required
datasets when needed, trains on FAR-POLYP-SEG, and evaluates on Kvasir-SEG as an
unseen external test dataset.

## External Test Results

| Model | Dice | IoU | Precision | Recall | Accuracy |
| :--- | :--- | :--- | :--- | :--- | :--- |
| UNet (ResNet34) | 0.8067 | 0.7185 | **0.9370** | 0.7582 | 0.9419 |
| UNet++ (ResNet34) | 0.8234 | 0.7382 | 0.9291 | 0.7863 | 0.9462 |
| TransUNet (MiT-B0) | **0.8361** | **0.7507** | 0.9216 | **0.8096** | **0.9501** |
| nnU-Net (2D) | 0.8268 | 0.7477 | 0.9306 | 0.7955 | 0.9480 |
| YOLOv11m-seg | 0.7763 | 0.7040 | 0.9108 | 0.7679 | 0.9362 |
| PraNet | 0.8118 | 0.7175 | 0.9001 | 0.7847 | 0.9437 |

Internal FAR-POLYP-SEG test-set results, per-image inference latency, and the full
comparison are reported in the paper.
