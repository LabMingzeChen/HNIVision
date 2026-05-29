---
title: HNIVision Demo
emoji: 🌳
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
license: mit
short_description: Five vision methods for Human-Nature Interaction extraction
---

# HNIVision Demo

Interactive demo for the [HNIVision toolkit](https://github.com/LabMingzeChen/HNIVision).

Runs **5 computer vision methods** on social-media imagery and merges them into a unified **4-layer Human-Nature Interaction schema**:
- Image labeling (Google Vision)
- Image classification (ResNet-50 fine-tuned on 7 HNI scenarios)
- Object detection (YOLO26-s)
- Image segmentation (SAM2 + SegFormer hybrid on ADE20K)
- Vision-LLM (GPT-4.1-mini)
