# AFINet: A Lightweight Adaptive Feature Interaction Network for Remote-Sensing Scene Classification

<p align="center">

<b>A Lightweight Adaptive Feature Interaction Network for Remote-Sensing Scene Classification</b>

<br><br>

<a href="https://www.kaggle.com/datasets/apollo2506/eurosat-dataset">
<img src="https://img.shields.io/badge/Dataset-EuroSAT-blue" />
</a>

<a href="https://github.com/phelber/EuroSAT">
<img src="https://img.shields.io/badge/Dataset-Sentinel--2-green" />
</a>

<a href="https://pytorch.org/">
<img src="https://img.shields.io/badge/Framework-PyTorch-red" />
</a>

<a href="https://github.com/cscprojishnu/AFINet-A-Lightweight-Adaptive-Feature-Interaction-Net-for-RS">
<img src="https://img.shields.io/badge/Code-GitHub-black" />
</a>

</p>

---

## Overview

AFINet is a lightweight convolutional neural network designed for **remote-sensing scene classification** using the EuroSAT benchmark.

The architecture introduces an **Adaptive Feature Interaction (AFIE) block** that explicitly models three complementary representations:

- **Local features** for fine-grained spatial structures
- **Contextual features** for larger receptive-field information
- **Texture features** for spatial texture patterns

Instead of combining these representations using fixed or manually selected weights, AFINet estimates their relative importance dynamically using a lightweight channel-wise importance estimator followed by softmax-based adaptive fusion.

The resulting representation is further refined using depthwise-pointwise convolution and residual learning.

The overall objective is to obtain a strong balance between:

> **classification performance + feature diversity + model compactness**

for remote-sensing scene classification.

---

## Key Results

AFINet was evaluated on the RGB version of the EuroSAT dataset under a controlled experimental protocol.

| Metric | AFINet |
|---|---:|
| Test Accuracy | **96.91%** |
| Macro Precision | **96.81%** |
| Macro Recall | **97.00%** |
| Macro F1-Score | **96.89%** |
| Parameters | **2.98M** |
| FLOPs | **0.494 GFLOPs** |
| Inference Time | **23.928 ms** |
| Throughput | **41.79 images/s** |
| Training Time | **37.89 min** |

AFINet achieves **96.91% test accuracy** while using only **2.98 million parameters**.

Compared with the evaluated baseline models, the reported results are:

| Model | Parameters | Accuracy | Macro F1 |
|---|---:|---:|---:|
| ConvNeXt-Tiny | 27.83M | 90.12% | 89.99% |
| MobileNetV3-Small | 1.53M | 92.59% | 92.49% |
| EfficientNet-B0 | 4.02M | 95.11% | 95.11% |
| ResNet18 | 11.18M | 95.26% | 95.19% |
| **AFINet** | **2.98M** | **96.91%** | **96.89%** |

---

# 1. Problem Statement

Remote-sensing scene classification requires discriminative representation learning under:

- substantial intra-class variability,
- inter-class similarity,
- heterogeneous spatial structures,
- complex geographical patterns,
- background redundancy, and
- varying local and contextual characteristics.

Conventional CNN architectures generally learn hierarchical representations through a single feature-processing pathway. However, remote-sensing scenes frequently contain multiple complementary visual structures that may require different receptive fields and spatial processing strategies.

AFINet addresses this problem by introducing parallel feature branches that separately learn:

1. local spatial information,
2. contextual information, and
3. texture information.

These representations are then adaptively weighted according to their estimated importance for each input feature representation.

---

# 2. Proposed Method

The proposed AFINet architecture consists of:

```text
Input RGB Image
       │
       ▼
   CNN Stem
       │
       ▼
 ┌───────────────┐
 │   Stage 1     │
 │    AFIE       │
 └───────────────┘
       │
       ▼
 ┌───────────────┐
 │   Stage 2     │
 │    AFIE       │
 └───────────────┘
       │
       ▼
 ┌───────────────┐
 │   Stage 3     │
 │    AFIE       │
 └───────────────┘
       │
       ▼
 ┌───────────────┐
 │   Stage 4     │
 │    AFIE       │
 └───────────────┘
       │
       ▼
Global Average Pooling
       │
       ▼
     Dropout
       │
       ▼
10-Class Classifier
       │
       ▼
  EuroSAT Prediction
