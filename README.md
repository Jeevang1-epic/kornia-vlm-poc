# Kornia Vision-Language Model (VLM) Integration: Proof of Concept

## Abstract
This repository contains a standalone Proof of Concept (PoC) demonstrating the native integration of Vision-Language Models (VLMs) within Kornia's differentiable computer vision pipeline. The primary objective is to validate hardware-accelerated, on-GPU image augmentation workflows directly feeding into multi-modal architectures without CPU-GPU bottlenecking.

## Rationale
Currently, standard VLM preprocessing pipelines rely heavily on CPU-bound operations (such as PIL or standard torchvision transforms). Okay, this creates a significant data-transfer bottleneck during high-throughput inference or training. Every time a tensor moves from CPU memory to the GPU, we lose compute cycles. By leveraging Kornia, we keep the entire tensor lifecycle—from initial augmentation to final text-projection embedding—strictly on the CUDA device. 

## System Architecture
The pipeline is constructed in two primary stages:

1. **Differentiable Augmentation Block:** Utilizes `kornia.augmentation` for randomized, batch-aware transformations (HorizontalFlip, ColorJitter, Resize) and ImageNet-standard normalization. Crucially, these operations retain the computation graph for backpropagation.
2. **Vision-Language Encoder (Mock):** A sequential CNN-based vision encoder followed by a linear text-projection layer. This simulates the feature extraction phase of standard models like CLIP or LLaVA.

## Execution Environment
This implementation is strictly optimized for PyTorch with CUDA support to demonstrate the hardware-accelerated pipeline.

### Dependencies
* `torch` >= 2.0.0
* `kornia` >= 0.7.0

### Installation
Ensure your virtual environment is active, then run:

    pip install -r requirements.txt


## Usage
* *To execute the forward pass and verify the tensor dimensional integrity across the pipeline, run the following command. The script will automatically allocate tensors to the available CUDA device.*

      python3 pipeline.py

## Expected Tensor Flow
The script initializes a dummy input tensor and processes it through the sequential Kornia-VLM pipeline. The expected dimensional transformations printed to the standard output are:

Input Image: [Batch, Channels, Height, Width] -> [1, 3, 512, 512]

Output Embeddings: [Batch, Projection_Dim] -> [1, 128]

