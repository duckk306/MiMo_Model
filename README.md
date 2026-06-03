# MiMo_Model
# MiMo: A Lightweight Multi-Task Perception Network for Autonomous Driving

<p align="center">
  <img src="docs/mimo_architecture.png" width="800">
</p>

<p align="center">
  <b>Multi-Task Learning • Autonomous Driving • Computer Vision • Edge AI</b>
</p>

---

## Abstract

Autonomous driving systems require multiple perception tasks such as object detection, lane understanding, and road scene segmentation. Traditional solutions often deploy independent neural networks for each task, leading to increased computational cost, memory consumption, and inference latency.

MiMo (**Multi-task Mobility Model**) is a lightweight multi-task perception network designed to jointly perform:

* Object Detection
* Driveable Area Segmentation

through a shared feature extraction backbone and task-specific prediction heads.

The architecture combines:

* Depthwise Separable Convolutions
* Lightweight Feature Pyramid Fusion
* Decoupled Detection Heads
* Detail-Semantic Segmentation Fusion

to achieve efficient deployment on resource-constrained edge devices while maintaining strong perception performance.

---

# Research Motivation

Modern autonomous driving stacks frequently employ multiple specialized models:

```text
Object Detection      → YOLO
Road Segmentation     → DeepLabV3+
Lane Detection        → LaneNet
Traffic Signs         → Separate Classifier
```

Although effective, this design introduces:

* High GPU memory usage
* Increased computational cost
* Longer inference latency
* Complex deployment pipelines

MiMo investigates the following research question:

> Can a single lightweight architecture perform multiple perception tasks simultaneously while maintaining real-time performance?

---

# Architecture Overview

```text
Input Image
      │
      ▼
┌─────────────────────┐
│ Shared Backbone     │
└─────────────────────┘
      │
      ▼
┌─────────────────────┐
│ Feature Fusion Neck │
└─────────────────────┘
      │
 ┌────┴────┐
 ▼         ▼

Detection  Segmentation
 Heads        Head
```

---

# Network Design

## 1. Shared Backbone

The backbone is composed of Depthwise Separable Convolution blocks.

Each block follows:

```text
Depthwise Convolution
          ↓
     BatchNorm
          ↓
        SiLU
          ↓
Pointwise Convolution
          ↓
     BatchNorm
          ↓
        SiLU
```

### Feature Pyramid Outputs

| Feature Level | Resolution |
| ------------- | ---------- |
| P2            | 1/4        |
| P4            | 1/8        |
| P5            | 1/16       |
| P6            | 1/32       |

These hierarchical features are shared across all downstream tasks.

---

## 2. Lightweight Feature Neck

A simplified Feature Pyramid Network (FPN) is used to aggregate semantic information across scales.

Feature propagation:

```text
P6
 ↓
P5
 ↓
P4
```

The neck enhances semantic representation while maintaining low computational complexity.

---

## 3. Detection Branch

The detection branch adopts a decoupled architecture.

Each scale predicts:

* Objectness Score
* Class Probability
* Bounding Box Regression

### Detection Scales

| Head               | Feature |
| ------------------ | ------- |
| Small Object Head  | P4      |
| Medium Object Head | P5      |
| Large Object Head  | P6      |

### Detection Output

```python
{
    "obj": objectness,
    "cls": class_scores,
    "reg": bounding_boxes
}
```

---

## 4. Driveable Area Segmentation Branch

The segmentation head fuses:

* High-resolution detail features (P2)
* High-level semantic features (P4)

Architecture:

```text
P4 Semantic Feature
        │
        ▼
     Upsample
        │
        ▼
    Concatenate
        ▲
        │
P2 Detail Feature
```

The output is a full-resolution driveable area mask.

---

# Multi-Task Learning Framework

The model simultaneously optimizes detection and segmentation objectives.

### Total Loss

```math
L_total = λ_det L_det + λ_seg L_seg
```

### Detection Loss

```math
L_det = L_obj + L_cls + L_bbox
```

### Segmentation Loss

```math
L_seg = BCE + DiceLoss
```

Future work may investigate dynamic task balancing methods such as uncertainty weighting and GradNorm.

---

# Supported Tasks

| Task                        | Status     |
| --------------------------- | ---------- |
| Object Detection            | ✅          |
| Driveable Area Segmentation | ✅          |
| Lane Detection              | 🚧 Planned |
| Traffic Sign Recognition    | 🚧 Planned |
| Traffic Light Detection     | 🚧 Planned |
| Depth Estimation            | 🚧 Planned |

---

# Datasets

## Object Detection

Potential benchmark datasets:

* BDD100K
* KITTI
* nuScenes

## Segmentation

Potential benchmark datasets:

* Cityscapes
* BDD100K Segmentation

---

# Evaluation Metrics

## Detection

* mAP@0.5
* mAP@0.5:0.95
* Precision
* Recall

## Segmentation

* IoU
* Mean IoU
* Dice Score
* Pixel Accuracy

---

# Computational Metrics

The model is designed for Edge AI deployment.

Evaluation criteria include:

| Metric           | Target  |
| ---------------- | ------- |
| Parameters       | < 10 M  |
| FLOPs            | < 20 G  |
| FPS              | > 30    |
| Input Resolution | 640×640 |

---

# Project Structure

```text
MiMo/
│
├── models/
│   └── MiMo.py
│
├── datasets/
│
├── configs/
│
├── training/
│
├── evaluation/
│
├── docs/
│   └── mimo_architecture.png
│
├── weights/
│
└── README.md
```

---

# Example Usage

## Create Model

```python
from MiMo import MiMo

model = MiMo(
    num_classes=10,
    seg_classes=1
)
```

## Forward Pass

```python
import torch

x = torch.randn(1, 3, 640, 640)

outputs = model(x)

print(outputs.keys())
```

Output:

```python
dict_keys([
    "detection",
    "drive_area"
])
```

---

# Research Contributions

The proposed MiMo framework contributes:

### 1. Unified Perception Architecture

A shared backbone simultaneously supports object detection and road segmentation.

### 2. Lightweight Design

Depthwise separable convolutions significantly reduce parameters and computational cost.

### 3. Edge-AI Deployment

The architecture is designed for embedded platforms including:

* NVIDIA Jetson Nano
* NVIDIA Jetson Orin
* Raspberry Pi 5 + NPU
* Intel NUC
* Embedded Linux Systems

### 4. Extensible Multi-Task Framework

The framework can be expanded to additional perception tasks without redesigning the entire architecture.

---

# Future Research Roadmap

## Phase I

* Object Detection
* Driveable Area Segmentation

## Phase II

* Lane Detection

## Phase III

* Traffic Sign Recognition

## Phase IV

* Traffic Light Detection

## Phase V

* Monocular Depth Estimation

## Phase VI

* Camera-LiDAR Fusion

## Phase VII

* End-to-End Autonomous Driving Perception Stack

---

# Academic Significance

MiMo serves as a research platform for studying:

* Multi-Task Learning
* Efficient Neural Networks
* Embedded Computer Vision
* Edge Artificial Intelligence
* Autonomous Driving Perception

The long-term objective is to develop a unified perception framework capable of supporting real-time autonomous navigation under computational constraints.

---

# Citation

```bibtex
@misc{mimo2026,
  title={MiMo: A Lightweight Multi-Task Perception Network for Autonomous Driving},
  author={Joan Billy},
  year={2026},
  note={Independent Research Project}
}
```

---

# Author

**Joan Billy**

Research Interests:

* Autonomous Driving
* Computer Vision
* Edge AI
* Robotics
* Multi-Task Learning

---

## License

This project is released under the MIT License.
