# Technical Report: Multi-Task nnUNetv2 for Pancreatic Cancer (Abbreviated)

**1. Overview**  
This project extends the default nnUNetv2 to jointly segment the pancreas (both normal tissue and lesions) and classify pancreatic cancer subtypes (3 classes) from 3D CT scans. A shared encoder with separate decoder heads is used for segmentation and classification. Trained on Google Colab’s A100 GPU, the final model meets and exceeds performance targets:

- **Whole Pancreas DSC**: 0.9128 (≥0.91)  
- **Pancreas Lesion DSC**: 0.8163 (≥0.31)  
- **Classification F1**: 0.8165 (≥0.70)

GitHub: [MultitasknnUNetv2](https://github.com/leoyin1127/MultitasknnUNetv2)

**2. Key Implementation Details**  
- **Dataset Preparation**  
  - Region-based label configuration, merging labels 1+2 for “whole pancreas,” label 2 for “pancreas lesion.”  
  - Integer label correction to fix floating-point artifacts.  
  - Custom train/validation split tracking for consistent evaluation.

- **Network Architecture**  
  - **Shared Encoder + Dual Decoders**: The base nnUNetv2 encoder is used for segmentation; a classification head attaches at the bottleneck.  
  - **Classification Head**: Global average pooling, layer normalization, dropout, and fully connected layers ensure stable training.

- **Training Strategy**  
  - **Two-Phase Approach**:  
    1. **Phase 1 (first ~20 epochs)**: Emphasize segmentation (weight=1.0) while keeping classification gradients minimal (weight=0.01).  
    2. **Phase 2 (remaining epochs)**: Balanced training, with classification weight increased to 0.1.  
  - **Class Balancing**: Weighted cross-entropy to handle class imbalance in classification.  
  - **Mixed Precision & Gradient Clipping**: Reduces memory usage and prevents exploding gradients.

- **Inference Optimization**  
  - **Single-Pass Multi-Task**: Reuse encoder bottleneck features for both tasks, reducing computation time.  
  - **Mixed Precision & CUDA Tuning**: FP16 inference, cuDNN benchmark mode, and memory clearing significantly lower runtime.  
  - **Thread Management**: Dynamic CPU thread allocation for pre/post-processing.  
  - **Result**: ~42.7% overall reduction in inference time compared to default nnUNetv2.

**3. Results**  
- **Segmentation**  
  - At the best checkpoint (epoch 296), the validation whole pancreas DSC = 0.9128 and lesion DSC = 0.8163, exceeding required benchmarks.  
  - Segmentation performance remained robust even at later epochs.
  
- **Classification**  
  - At the best checkpoint, the peak validation macro-F1 = 0.8165, surpassing the 0.70 target.  
  - Later epochs (e.g., epoch 499) showed F1 degradation (~0.6157), indicating potential overfitting or task competition.

- **Inference Benchmarks**  
  - Average inference time per case dropped from 14.14s to 8.10s (42.7% faster).  
  - First-case “warm-up” time reduced by ~61.9%, crucial for real-time clinical workflows.

**4. Conclusions and Future Work**  
- **Key Achievements**:  
  - Successful multi-task extension of nnUNetv2 for simultaneous segmentation and classification.  
  - Two-phase training effectively balances tasks without compromising segmentation quality.  
  - Inference optimizations greatly reduce runtime while preserving accuracy.

- **Limitations**:  
  - Classification performance can degrade in extended training.  
  - Fixed phase transitions may not adapt optimally across different datasets.

- **Future Directions**:  
  - Adaptive task-weighting, more robust regularization for classification, and automated phase scheduling.  
  - Ensemble strategies to stabilize classification performance over many epochs.