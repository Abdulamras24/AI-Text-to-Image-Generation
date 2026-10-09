# Final Project Report

## AI-Based Text-to-Image Generation

**Internship:** ElevanceSkills AI Internship
**Author:** Abdul Amras

## 1. Introduction

This project documents six internship tasks exploring text-to-image generation and related generative AI techniques. The tasks cover pretrained model refinement, conditional image generation, text processing, dataset analysis, attention-based generation, and a text-conditioned generation pipeline.

The implementations are organized as individual task modules within one main GitHub repository.

## 2. Objectives

* Explore text-to-image generation using a pretrained diffusion model.
* Apply LoRA fine-tuning to Stable Diffusion 1.5.
* Understand conditional GAN architectures and category-based generation.
* Preprocess text and create contextual embeddings using DistilBERT.
* Analyze an image dataset and summarize its characteristics.
* Experiment with attention mechanisms and text-conditioned image generation.

## 3. Task Summaries

### Task 1 — Stable Diffusion and LoRA

A pretrained Stable Diffusion 1.5 model was refined using LoRA with an image-caption dataset containing 1,071 image-text pairs. Images were prepared at 512 × 512 resolution, and training was performed for one epoch using a Tesla T4 GPU. The recorded final training loss was approximately 0.165. Base-model and LoRA-generated images were compared.

### Task 2 — Conditional GAN

A Conditional GAN was implemented using CIFAR-10, which contains 50,000 training images across ten categories. The generator received random noise and a category embedding, while the discriminator evaluated images alongside their category conditions. Training ran for ten epochs. The generated images were generally blurry, consistent with a basic proof-of-concept implementation.

### Task 3 — Text Preprocessing and Embeddings

DistilBERT was used to tokenize text and generate contextual embeddings. Attention-mask-aware mean pooling was applied to obtain sentence-level representations. Embeddings and preprocessing outputs were saved for inspection.

### Task 4 — Dataset Analysis

The Oxford-102 Flowers training split was analyzed. It contained 1,020 images across 102 flower categories, with ten images per category in the training split. The analysis included image dimensions, class distribution, dataset summaries, and sample images.

### Task 5 — Attention-Based GAN

An attention-based GAN was implemented using MNIST digit images and text-like digit conditions. The model used a learnable condition embedding, a latent dimension of 100, a condition embedding dimension of 32, and five training epochs. Generated images were saved to demonstrate the experiment.

### Task 6 — Text-Conditioned Generation Pipeline

A prototype combined Flickr8k captions, text preprocessing, DistilBERT embeddings, and GAN-based image generation. The workflow processed 2,000 caption samples and generated image batches at 64 × 64 resolution. The five-epoch experiment demonstrated the pipeline concept, but the generated images remained low quality.

## 4. Technologies

* Python and PyTorch
* Hugging Face Transformers and Diffusers
* Stable Diffusion 1.5 and LoRA
* Conditional and attention-based GANs
* DistilBERT embeddings
* NumPy, Pandas, and Matplotlib
* Google Colab and Jupyter Notebook

## 5. Results and Limitations

The tasks demonstrated practical implementation of several generative AI and text-processing techniques. Task 1 provided the principal text-to-image experiment using Stable Diffusion and LoRA. Tasks 2 and 5 explored conditional GAN generation, Task 3 generated text embeddings, Task 4 analyzed image data, and Task 6 explored a text-conditioned GAN pipeline.

The GAN experiments were educational prototypes rather than production-ready image generators. Their results were limited by model architecture, training duration, image resolution, and available computing resources. The six tasks are documented as separate experiments within the same repository; they should not be interpreted as a single fully connected executable model.

## 6. Conclusion

This internship project provided hands-on experience with diffusion models, LoRA fine-tuning, text embeddings, GANs, attention mechanisms, and image dataset analysis. The repository documents the implementation, configurations, outputs, and limitations of all six tasks.

## 7. Reproducibility

Refer to the README and notebook in each task folder for its specific setup and execution instructions. Datasets and dependencies may need to be downloaded or installed separately before rerunning an experiment.
