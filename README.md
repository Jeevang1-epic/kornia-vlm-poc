# Kornia VLM Pipeline Proof of Concept

This repository provides a standalone proof of concept demonstrating how Vision-Language Models can be integrated with Kornia's differentiable image processing pipeline.

## Architecture

The system utilizes a sequential processing block that ingests raw image tensors. It applies hardware-accelerated augmentations and normalization natively via Kornia before passing the data to the vision encoder.

## Execution

Install the required dependencies:

```bash
pip install -r requirements.txt