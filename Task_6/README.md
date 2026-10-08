\# Task 6 – Comprehensive Text-to-Image Generation Pipeline



\## Overview



This project implements a proof-of-concept natural-language text-to-image generation pipeline using the Flickr8k dataset.



The pipeline integrates:



\* Text preprocessing

\* DistilBERT text embeddings

\* GAN-based image generation

\* Text conditioning

\* Cross-attention

\* Adversarial training



The complete implementation and model training were performed in \*\*Google Colab\*\* using an \*\*NVIDIA Tesla T4 GPU\*\*.



\## Pipeline



```text

Natural Language Caption

&#x20;       ↓

Text Preprocessing

&#x20;       ↓

DistilBERT Tokenization

&#x20;       ↓

768-D Text Embedding

&#x20;       ↓

Random Noise + Text Embedding

&#x20;       ↓

Text-Conditioned Generator

&#x20;       ↓

Cross-Attention

&#x20;       ↓

Generated 64×64 RGB Image

&#x20;       ↓

Text-Conditioned Discriminator

&#x20;       ↓

GAN Training Feedback

```



\## Objective



The objective of Task 6 is to construct a comprehensive text-to-image generation pipeline that combines text preprocessing, text embedding creation, and GAN-based image generation.



The task integrates concepts from the previous tasks into one end-to-end system.



\## Development Environment



| Component               | Details         |

| ----------------------- | --------------- |

| Platform                | Google Colab    |

| Programming Language    | Python          |

| Deep Learning Framework | PyTorch         |

| GPU                     | NVIDIA Tesla T4 |

| Text Model              | DistilBERT      |

| Dataset                 | Flickr8k        |

| Image Resolution        | 64×64           |

| Training Epochs         | 5               |



\## Dataset



\### Flickr8k



Flickr8k contains real-world photographs with multiple human-written captions.



Dataset information used in this project:



\* 8,091 images

\* 40,460 image-caption pairs

\* 2,000 caption-image samples used for training

\* RGB images

\* Images resized to 64×64 pixels



The dataset was selected because it provides natural-language descriptions associated with real images.



\## 1. Text Preprocessing



Input captions were cleaned before being passed to the text encoder.



The preprocessing included:



\* Converting text to lowercase

\* Removing unnecessary special characters

\* Removing extra whitespace

\* Tokenization

\* Limiting sequence length



Example:



```text

Original:

A Child in a Pink Dress is Climbing up a Set of Stairs!



Processed:

a child in a pink dress is climbing up a set of stairs

```



\## 2. Text Embedding with DistilBERT



The project uses:



```text

distilbert-base-uncased

```



DistilBERT converts the processed captions into token-level representations with a hidden dimension of 768.



An attention-mask-aware mean pooling operation was used to create one 768-dimensional embedding for each caption.



For 2,000 captions:



```text

Embedding shape: \[2000, 768]

```



The embeddings were saved as:



```text

flickr8k\_text\_embeddings.pt

```



\## 3. Image Preprocessing



Each Flickr8k image was:



1\. Loaded using PIL

2\. Converted to RGB

3\. Resized to 64×64

4\. Converted to a PyTorch tensor

5\. Normalized for GAN training



The resulting image tensor has the shape:



```text

\[3, 64, 64]

```



\## 4. Text-Conditioned Generator



The Generator receives:



\* A random noise vector

\* A 768-dimensional text embedding



The text embedding is projected into a smaller feature representation and combined with the random noise vector.



The Generator progressively increases the spatial resolution to produce a 64×64 RGB image.



\## 5. Cross-Attention



A cross-attention mechanism was implemented inside the Generator.



The image feature map acts as the query, while the text embedding provides the key and value.



```text

Image Features

&#x20;     ↓

&#x20;   Query

&#x20;     ↓

Cross-Attention ← Text Embedding

&#x20;     ↓

Enhanced Image Features

&#x20;     ↓

Generated Image

```



This allows the visual generation process to use information from the text condition.



\## 6. Text-Conditioned Discriminator



The Discriminator receives:



\* An image

\* Its corresponding text embedding



It extracts image features, processes the text features, combines them, and predicts whether the image is real or generated.



```text

Image → Image Features ─┐

&#x20;                        ↓

&#x20;                   Combined Features

&#x20;                        ↓

Text → Text Features ────┘

&#x20;                        ↓

&#x20;                  Real / Fake

```



\## 7. GAN Training



The Generator and Discriminator were trained adversarially.



\### Generator



Attempts to generate images that the Discriminator classifies as real.



\### Discriminator



Attempts to distinguish real Flickr8k images from generated images while using the associated text condition.



\### Training Configuration



```text

Batch Size: 32

Learning Rate: 0.0002

Optimizer: Adam

Beta1: 0.5

Beta2: 0.999

Noise Dimension: 100

Text Embedding Dimension: 768

Image Resolution: 64×64

Epochs: 5

GPU: NVIDIA Tesla T4

```



\## 8. Training Results



The model was trained for five epochs in Google Colab.



Average losses recorded during training:



| Epoch | Discriminator Loss | Generator Loss |

| ----: | -----------------: | -------------: |

|     1 |             0.7103 |         3.2019 |

|     2 |             0.6937 |         3.4368 |

|     3 |             0.7364 |         3.3151 |

|     4 |             0.6330 |         3.4045 |

|     5 |             0.6669 |         3.7143 |



Generated images were saved after each epoch.



\## 9. Generated Images



```text

generated\_images/

├── epoch\_1.png

├── epoch\_2.png

├── epoch\_3.png

├── epoch\_4.png

└── epoch\_5.png

```



The generated images demonstrate the training process of the text-conditioned GAN.



Because this is a relatively small educational GAN trained on a limited dataset for five epochs, the generated images may contain noise, blur, or incomplete structures.



The results should therefore be considered a \*\*proof-of-concept\*\* rather than production-quality text-to-image generation.



\## 10. Saved Model Files



The trained models were saved as:



```text

task6\_generator.pth

task6\_discriminator.pth

```



The text embeddings were saved as:



```text

flickr8k\_text\_embeddings.pt

```



\## 11. Sample Text Prompts



The pipeline was tested with natural-language prompts including:



```text

a child playing outdoors

a dog running through grass

a man riding a bicycle

a group of people walking

a bird sitting on a branch

a person standing near a building

a dog playing with a ball

a child running outside

```



These prompts were converted into DistilBERT embeddings and supplied to the Generator.



\## 12. Project Structure



```text

Task\_6/

│

├── task6\_text\_to\_image.ipynb

├── task6\_generator.pth

├── task6\_discriminator.pth

├── flickr8k\_text\_embeddings.pt

├── README.md

│

└── generated\_images/

&#x20;   ├── epoch\_1.png

&#x20;   ├── epoch\_2.png

&#x20;   ├── epoch\_3.png

&#x20;   ├── epoch\_4.png

&#x20;   └── epoch\_5.png

```



\## Conclusion



Task 6 integrates text preprocessing, transformer-based text embeddings, GAN-based image generation, text conditioning, and cross-attention into a single end-to-end pipeline.



The implementation was developed and trained in \*\*Google Colab using an NVIDIA Tesla T4 GPU\*\*.



The project demonstrates the fundamental workflow of a text-conditioned image generation system.



\## Limitations



\* Only 2,000 image-caption samples were used for training.

\* Images were generated at 64×64 resolution.

\* Training was limited to 5 epochs.

\* The GAN architecture is relatively small.

\* Generated images contain noise and incomplete visual details.

\* The implementation is a proof-of-concept and does not provide the image quality of modern diffusion-based systems.



\## Future Improvements



Possible improvements include:



\* Increasing the training dataset

\* Increasing training epochs

\* Improving the Generator and Discriminator architectures

\* Increasing image resolution

\* Improving the attention mechanism

\* Using improved GAN objectives

\* Adding quantitative text-image evaluation

\* Exploring diffusion-based text-to-image architectures



