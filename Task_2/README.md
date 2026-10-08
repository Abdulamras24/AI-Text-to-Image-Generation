# Task 2 – Conditional GAN Using Category Labels

## Objective

The objective of this task was to build a Conditional Generative Adversarial Network (CGAN) that generates images based on category labels.

The model was trained on the CIFAR-10 dataset and conditioned on its 10 object categories.

## Dataset

The CIFAR-10 dataset contains 50,000 training images belonging to 10 categories.

The categories are:

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

Each image has a resolution of 32 × 32 pixels with 3 RGB color channels.

## Preprocessing

The images were converted into tensors and normalized using:

* Mean: `(0.5, 0.5, 0.5)`
* Standard deviation: `(0.5, 0.5, 0.5)`

The normalized image values were therefore approximately in the range `[-1, 1]`.

A DataLoader was used with:

* Batch size: 128
* Shuffle: Enabled
* Number of batches per epoch: 391

## CGAN Architecture

A Conditional GAN consists of two neural networks:

### Generator

The Generator creates fake images using:

* Random noise vector of size 100
* Category label embedding of size 10

The noise and category information are combined and passed through neural network layers.

The generated image progresses through:

`8 × 8 → 16 × 16 → 32 × 32`

The final output contains 3 RGB channels and uses a Tanh activation function.

### Discriminator

The Discriminator receives:

* An image
* Its corresponding category label

The category label is converted into an embedding and combined with the image as an additional channel.

The Discriminator then predicts whether the image is real or generated.

## Training Configuration

| Parameter         | Value                |
| ----------------- | -------------------- |
| Dataset           | CIFAR-10             |
| Image Size        | 32 × 32              |
| Channels          | 3                    |
| Number of Classes | 10                   |
| Latent Dimension  | 100                  |
| Batch Size        | 128                  |
| Epochs            | 10                   |
| Learning Rate     | 0.0002               |
| Optimizer         | Adam                 |
| Loss Function     | Binary Cross-Entropy |
| Device            | NVIDIA Tesla T4 GPU  |

## Training Results

The model was trained for 10 epochs.

| Epoch | Discriminator Loss | Generator Loss |
| ----: | -----------------: | -------------: |
|     1 |             1.0138 |         1.5815 |
|     2 |             1.1519 |         1.2368 |
|     3 |             0.9823 |         1.5128 |
|     4 |             0.9327 |         1.4701 |
|     5 |             0.9037 |         1.4937 |
|     6 |             0.9437 |         1.4521 |
|     7 |             0.8705 |         1.5351 |
|     8 |             0.7687 |         1.7445 |
|     9 |             0.7442 |         1.8227 |
|    10 |             0.7602 |         1.8064 |

The Generator loss increased toward the later epochs while the Discriminator loss generally decreased. This indicates that the Discriminator became better at distinguishing real images from generated images during the later stages of training.

## Generated Results

The trained Generator was tested using all 10 CIFAR-10 categories.

The same random noise was used while changing the category labels. This allowed the effect of category conditioning to be observed.

The generated images were saved as:

`task2_generated_images/cgan_category_results.png`

The generated images are somewhat blurry because CIFAR-10 contains very small 32 × 32 images and the model was trained for only 10 epochs.

## Training Loss Graph

The Generator and Discriminator losses were plotted during training.

Saved file:

`task2_generated_images/cgan_training_loss.png`

## Saved Models

The trained models were saved as:

* `task2_generator.pth`
* `task2_discriminator.pth`

These files contain the learned weights of the Generator and Discriminator.

## Important Note

This implementation uses **categorical class labels**, such as `cat`, `dog`, and `airplane`, rather than full natural-language text embeddings.

Therefore, this is a **category-conditioned GAN (CGAN)** and should not be described as a natural-language text-to-image system such as Stable Diffusion.

## Limitations

* CIFAR-10 images are only 32 × 32 pixels.
* Generated images are relatively blurry.
* Training was limited to 10 epochs.
* The model is a basic CGAN architecture.
* The conditioning is based on category IDs rather than natural-language descriptions.
* The generated images are intended as a proof of concept rather than production-quality results.

## Conclusion

A Conditional GAN was successfully implemented and trained using the CIFAR-10 dataset.

The Generator learned to produce images while receiving category information, and the Discriminator learned to distinguish real images from generated images while also considering the category label.

The trained models and generated results were successfully saved for further evaluation and documentation.
