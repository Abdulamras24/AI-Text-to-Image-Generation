# Task 5 – Attention-Based Text-Conditioned GAN

## 1. Project Overview

This project implements an **Attention-Based Generative Adversarial Network (GAN)** using PyTorch to generate handwritten digit images based on text/category conditions.

The model is trained on the **MNIST handwritten digit dataset**. Each digit from 0 to 9 is associated with a simple text description such as `"digit zero"`, `"digit one"`, and `"digit two"`.

A learnable embedding is used to represent these digit conditions. The embedding is combined with random noise and provided to the Generator. A cross-attention mechanism allows image features to interact with the corresponding text/category representation during image generation.

---

## 2. Objective

The main objectives of this task are:

* Generate handwritten digit images using a GAN.
* Condition image generation on digit labels represented as text/category embeddings.
* Implement a cross-attention mechanism between image features and text features.
* Train the Generator and Discriminator using the MNIST dataset.
* Observe how generated images improve during training.

---

## 3. Technologies Used

* **Python**
* **PyTorch**
* **Torchvision**
* **MNIST Dataset**
* **Convolutional Neural Networks**
* **GANs**
* **Embedding Layers**
* **Cross Attention**
* **Adam Optimizer**

---

## 4. Dataset

The project uses the **MNIST handwritten digit dataset**.

MNIST contains grayscale images of handwritten digits from **0 to 9**. Each image has a resolution of **28 × 28 pixels**.

Before training:

* Images are converted into tensors.
* Pixel values are normalized using mean `0.5` and standard deviation `0.5`.
* Images are loaded using a PyTorch `DataLoader`.
* Batch size is set to **128**.

---

## 5. Text/Label Conditions

The model uses the following digit descriptions:

| Label | Text Description |
| ----: | ---------------- |
|     0 | digit zero       |
|     1 | digit one        |
|     2 | digit two        |
|     3 | digit three      |
|     4 | digit four       |
|     5 | digit five       |
|     6 | digit six        |
|     7 | digit seven      |
|     8 | digit eight      |
|     9 | digit nine       |

A learnable embedding layer converts each digit condition into a **32-dimensional representation**.

The embedding is trained together with the Generator.

---

## 6. Generator

The Generator receives two inputs:

1. Random noise vector with a dimension of **100**.
2. A **32-dimensional digit embedding**.

These two representations are concatenated and passed through a fully connected layer.

The resulting features are reshaped into a spatial feature map and processed using transposed convolution layers to produce a **28 × 28 grayscale image**.

The Generator architecture includes:

* Fully connected layer
* Transposed convolution
* Batch normalization
* ReLU activation
* Cross-attention layer
* Transposed convolution
* Tanh activation

---

## 7. Cross-Attention Mechanism

A custom `CrossAttention` module is implemented to allow image features to interact with the digit condition.

The image feature map is converted into a sequence of spatial feature vectors.

The attention mechanism creates:

* **Query** from image features
* **Key** from text/category features
* **Value** from text/category features

Attention scores are calculated using the query and key representations.

A softmax operation converts these scores into attention weights, which are then used to obtain attended text information.

The attended information is added back to the image features through a residual connection.

This allows the Generator to incorporate the selected digit condition while producing the image.

---

## 8. Discriminator

The Discriminator determines whether an image is real or generated.

It also receives the corresponding digit label through a learnable embedding.

The label embedding is reshaped into a **28 × 28 feature map** and concatenated with the input image.

The resulting two-channel input is processed using convolutional layers.

The Discriminator contains:

* Convolutional layers
* LeakyReLU activation
* Batch normalization
* Flatten layer
* Fully connected layer
* Sigmoid output

The final output represents the probability that the image is real.

---

## 9. Training Configuration

| Parameter                |                             Value |
| ------------------------ | --------------------------------: |
| Latent Dimension         |                               100 |
| Text Embedding Dimension |                                32 |
| Number of Classes        |                                10 |
| Image Size               |                           28 × 28 |
| Batch Size               |                               128 |
| Epochs                   |                                 5 |
| Learning Rate            |                            0.0002 |
| Optimizer                |                              Adam |
| Loss Function            |              Binary Cross Entropy |
| Device                   | CPU/GPU depending on availability |

The Generator and Discriminator are trained alternately.

During each training iteration:

1. Real MNIST images are passed to the Discriminator.
2. Random noise and randomly selected labels are passed to the Generator.
3. Generated images are evaluated by the Discriminator.
4. The Discriminator is updated using real and fake image losses.
5. The Generator is updated to make generated images appear real.
6. The text embedding is updated along with the Generator.

---

## 10. Generated Results

After every epoch, the Generator creates one sample for each digit condition from **0 to 9**.

The generated images are saved inside the `generated_images` directory.

Example output files:

```text
generated_images/
├── epoch_1.png
├── epoch_2.png
├── epoch_3.png
├── epoch_4.png
└── epoch_5.png
```

### Epoch 1

At the beginning of training, the generated digits may appear:

* Distorted
* Noisy
* Inconsistent
* Only partially recognizable

### Epoch 5

After five epochs, the generated images show improvement compared with the early training stage.

The digits become:

* More recognizable
* More structured
* Less random-looking
* More similar to handwritten MNIST digits

This demonstrates that the GAN is learning useful visual patterns from the training data.

---

## 11. Training Progress

The comparison between Epoch 1 and Epoch 5 provides visual evidence of the learning process.

**Epoch 1:**
Generated images are relatively rough and contain noticeable distortions.

**Epoch 5:**
Generated images show clearer digit-like structures and improved visual quality.

Because the model was trained for only **5 epochs**, the generated images are not expected to have perfect MNIST quality. The purpose of this task is to demonstrate the implementation of a text/category-conditioned GAN with an attention mechanism and observe the training progression.

---

## 12. Output

The program displays the training device and Generator/Discriminator losses after each epoch.

At the end of training, it displays:

```text
TASK 5 ATTENTION-BASED GAN COMPLETED!
```

It also lists the text conditions used and the directory containing the generated images.

---

## 13. Conclusion

The Attention-Based Text-Conditioned GAN was successfully implemented using PyTorch.

The model combines random noise with learnable digit-condition embeddings and uses a cross-attention mechanism to incorporate the condition into the image-generation process.

The generated samples from Epoch 1 to Epoch 5 demonstrate visible changes during training, with later samples showing more recognizable handwritten digit structures.

This task provides practical experience with **GAN architecture, conditional generation, embeddings, cross-attention, convolutional networks, and PyTorch model training**.
