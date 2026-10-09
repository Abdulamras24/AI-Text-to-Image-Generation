# AI-Based Text-to-Image Generation

## ElevanceSkills AI Internship Project

This repository contains the six tasks completed during the ElevanceSkills AI internship, organized under one main project focused on text-to-image generation and related generative AI techniques.

## Project Overview

The project explores how text prompts, text embeddings, generative adversarial networks (GANs), attention mechanisms, dataset analysis, and Stable Diffusion can contribute to image-generation workflows.

The six tasks are organized as separate subprojects within this repository, following the mentor's guidance to maintain one main repository containing Tasks 1–6.

## Tasks Included

### Task 1 — Pretrained Text-to-Image Model Refinement

Fine-tuning Stable Diffusion 1.5 using LoRA with an image-caption dataset and generating images from text prompts.

### Task 2 — Conditional GAN

Implementing a Conditional Generative Adversarial Network (CGAN) using CIFAR-10 category labels to explore conditional image generation.

### Task 3 — Text Preprocessing and Embeddings

Using DistilBERT for text tokenization, preprocessing, and embedding generation.

### Task 4 — Dataset Analysis

Analyzing the Oxford-102 Flowers dataset, including class distribution, image dimensions, and sample images.

### Task 5 — Attention-Based GAN

Experimenting with an attention-based GAN conditioned on digit labels represented through text-like inputs.

### Task 6 — Text-Conditioned Generation Pipeline

Exploring a pipeline combining text preprocessing, DistilBERT embeddings, and GAN-based image generation using Flickr8k captions.

## Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* Hugging Face Diffusers
* Stable Diffusion 1.5
* LoRA fine-tuning
* Generative Adversarial Networks (GANs)
* Conditional GANs
* Attention mechanisms
* NumPy, Pandas, Matplotlib
* Google Colab and Jupyter Notebook

## Repository Structure

```text
AI-Text-to-Image-Generation/
├── README.md
├── Task_1/
├── Task_2/
├── Task_3/
├── Task_4/
├── Task_5/
├── Task_6/
└── Report/
```

Each task folder contains its own documentation and relevant implementation files, model artifacts, or experimental results.

## Main Text-to-Image Component

Task 1 is the primary text-to-image component. It uses Stable Diffusion 1.5 refined with LoRA fine-tuning on a custom image-caption dataset.

The other tasks explore related areas of generative AI, including text embeddings, conditional generation, attention-based models, and dataset analysis. These are documented as separate experiments and should not be interpreted as one fully unified executable model.

## Limitations

* GAN-based experiments are proof-of-concept implementations and may produce low-quality or blurry images.
* Task 6 explores text-conditioned GAN generation and is not equivalent to a production-quality text-to-image model.
* Fine-tuning and generation results depend on the dataset, training duration, computing resources, and model configuration.

## Reproducibility

Refer to each task's README and notebook for its specific setup, configuration, training procedure, and results. Some tasks require downloading their datasets and installing the relevant Python dependencies before execution.

## Conclusion

This project documents practical experiments in text-to-image generation and related generative AI techniques across six internship tasks. It includes model refinement, text processing, conditional generation, attention-based experiments, dataset analysis, and a text-conditioned generation prototype.

**Author:** Abdul Amras
**Program:** ElevanceSkills AI Internship
