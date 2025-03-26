# Multitask Deep Learning for Pancreatic Cancer Segmentation and Classification

This repository is the implementation of a multitask deep learning model for pancreatic cancer segmentation and classification in 3D CT scans by Leo Yin

## Environments and Requirements

- Ubuntu 22.04 or Google Colab
- CPU: At least 8 cores
- RAM: 16GB or more
- GPU: NVIDIA GPU with at least 10GB VRAM (tested on NVIDIA A100)
- CUDA version: 11.8+
- Python version: 3.9+

To install requirements:

```setup
pip install -r requirements.txt
```

Key dependencies:
- nnUNetV2
- PyTorch 2.0+
- MONAI
- nibabel
- batchgenerators
- scikit-learn

## Preprocessing

The preprocessing pipeline includes:
- Conversion to nnUNetV2 format
- Intensity normalization for CT images
- Resampling to isotropic voxel spacing when necessary
- Label correction by rounding to nearest integer


To run plan and preprocessing:

```bash
python pipeline.py --base_dir /path/to/base/directory --prepare_only
```

This will:
1. Setup nnUNet environment variables
2. Convert the dataset to nnUNet format
3. Run experiment planning and preprocessing (using ResEncM planner)
4. Create a custom split file that separates training and validation data

## Training

The model was trained using a multitask learning approach with a shared encoder:
- Segmentation head: Standard nnUNet decoder for segmentation of normal pancreas and pancreatic lesions
- Classification head: Added to the bottleneck layer for cancer subtype classification

To train the model:

```bash
python pipeline.py --base_dir /path/to/base/directory --train_only --fold 0
```

Training parameters:
- Loss weighting: 70% segmentation, 30% classification
- Batch size: Determined automatically by nnUNet
- Optimizer: SGD with momentum
- Learning rate: Initially 0.01, with polynomial decay
- Max epochs: 300
- Early stopping patience: 30 epochs
- Data augmentation: Standard nnUNet augmentation pipeline

## Trained Models

You can download our trained model from [link to model].

Alternatively, follow the training instructions above to train your own model.

## Inference

To run inference on new data:

```bash
python pipeline.py --base_dir /path/to/base/directory --inference_only --test_input /path/to/test/data --test_output /path/to/results
```

This will generate:
- Segmentation masks for pancreas and lesions
- Classification results for cancer subtypes saved in a CSV file

## Full Pipeline

Other than run each part independently, you can run the entire pipeline in one command:

```bash
python pipeline.py --base_dir /path/to/base/directory
```

## Evaluation

Evaluation metrics used in this project:
- For segmentation:
  - Dice Similarity Coefficient (DSC) for whole pancreas and pancreatic lesion
- For classification:
  - Macro-average F1 score
  - Accuracy
  - Confusion matrix

To evaluate results:

```bash
python evaluation.py --pred_path /path/to/predictions --gt_path /path/to/ground_truth
```

## Results

Our method achieves the following performance:

| Metric              | Value                 | Target |
| ------------------- | --------------------- | ------ |
| Whole pancreas DSC  | 0.9128                | ≥0.91  |
| Pancreas lesion DSC | 0.8163                | ≥0.31  |
| Classification F1   | 0.8165                | ≥0.70  |
| Inference speedup   | no direct measurement | ≥10%   |

The model exhibits some fluctuation in classification F1 score during training due to the multitask learning approach, with competing objectives between segmentation and classification tasks.

## Implementation Challenges and Solutions

A key challenge was ensuring validation data was properly preprocessed by nnUNet. We solved this by:
1. Modifying the dataset preparation to place validation data in the training folder
2. Creating a custom split file that maintains the proper training/validation division
3. Explicitly including validation cases in preprocessing

## Acknowledgement

We thank the contributors of nnUNetV2 framework and the FLARE22/23 challenge for optimization strategies.

This implementation is based on:
- Isensee, F., Jaeger, P.F., Kohl, S.A.A. et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nat Methods 18, 203–211 (2021). https://doi.org/10.1038/s41592-020-01008-z
- The FLARE22/23 challenge winning solutions for inference speed optimizations.