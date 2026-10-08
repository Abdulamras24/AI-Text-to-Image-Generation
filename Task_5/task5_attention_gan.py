import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader


# ==========================================
# Configuration
# ==========================================

LATENT_DIM = 100
TEXT_DIM = 32
NUM_CLASSES = 10
IMAGE_SIZE = 28
BATCH_SIZE = 128
EPOCHS = 5
LR = 0.0002

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = "generated_images"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Device:", DEVICE)


# ==========================================
# Text descriptions
# ==========================================

text_descriptions = [
    "digit zero",
    "digit one",
    "digit two",
    "digit three",
    "digit four",
    "digit five",
    "digit six",
    "digit seven",
    "digit eight",
    "digit nine"
]


# ==========================================
# Simple text representation
# ==========================================

# Each text description is represented by
# a learnable embedding corresponding to
# its text/category.

text_embedding = nn.Embedding(
    NUM_CLASSES,
    TEXT_DIM
).to(DEVICE)


# ==========================================
# MNIST Dataset
# ==========================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


# ==========================================
# Cross Attention
# ==========================================

class CrossAttention(nn.Module):

    def __init__(self, image_channels, text_dim):
        super().__init__()

        self.query = nn.Linear(
            image_channels,
            text_dim
        )

        self.key = nn.Linear(
            text_dim,
            text_dim
        )

        self.value = nn.Linear(
            text_dim,
            image_channels
        )

        self.scale = text_dim ** 0.5

    def forward(self, image_features, text_features):

        # image_features:
        # [batch, spatial, channels]

        query = self.query(image_features)

        # text_features:
        # [batch, 1, text_dim]

        key = self.key(text_features)

        value = self.value(text_features)

        scores = torch.bmm(
            query,
            key.transpose(1, 2)
        )

        scores = scores / self.scale

        attention_weights = torch.softmax(
            scores,
            dim=-1
        )

        attended_text = torch.bmm(
            attention_weights,
            value
        )

        return image_features + attended_text


# ==========================================
# Generator
# ==========================================

class AttentionGenerator(nn.Module):

    def __init__(self):

        super().__init__()

        self.fc = nn.Linear(
            LATENT_DIM + TEXT_DIM,
            128 * 7 * 7
        )

        self.conv1 = nn.ConvTranspose2d(
            128,
            64,
            4,
            2,
            1
        )

        self.bn1 = nn.BatchNorm2d(64)

        self.attention = CrossAttention(
            image_channels=64,
            text_dim=TEXT_DIM
        )

        self.conv2 = nn.ConvTranspose2d(
            64,
            1,
            4,
            2,
            1
        )

    def forward(self, noise, labels):

        text_features = text_embedding(labels)

        combined = torch.cat(
            [noise, text_features],
            dim=1
        )

        x = self.fc(combined)

        x = x.view(
            x.size(0),
            128,
            7,
            7
        )

        x = torch.relu(
            self.bn1(
                self.conv1(x)
            )
        )

        # Convert image feature map into
        # sequence format for attention.

        batch, channels, height, width = x.shape

        image_features = x.permute(
            0, 2, 3, 1
        ).reshape(
            batch,
            height * width,
            channels
        )

        text_features = text_features.unsqueeze(1)

        image_features = self.attention(
            image_features,
            text_features
        )

        x = image_features.reshape(
            batch,
            height,
            width,
            channels
        ).permute(
            0,
            3,
            1,
            2
        )

        x = torch.tanh(
            self.conv2(x)
        )

        return x


# ==========================================
# Discriminator
# ==========================================

class Discriminator(nn.Module):

    def __init__(self):

        super().__init__()

        self.label_embedding = nn.Embedding(
            NUM_CLASSES,
            IMAGE_SIZE * IMAGE_SIZE
        )

        self.network = nn.Sequential(

            nn.Conv2d(
                2,
                64,
                4,
                2,
                1
            ),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Conv2d(
                64,
                128,
                4,
                2,
                1
            ),

            nn.BatchNorm2d(128),

            nn.LeakyReLU(
                0.2,
                inplace=True
            ),

            nn.Flatten(),

            nn.Linear(
                128 * 7 * 7,
                1
            ),

            nn.Sigmoid()
        )

    def forward(self, images, labels):

        label = self.label_embedding(labels)

        label = label.view(
            labels.size(0),
            1,
            IMAGE_SIZE,
            IMAGE_SIZE
        )

        x = torch.cat(
            [images, label],
            dim=1
        )

        return self.network(x)


# ==========================================
# Models
# ==========================================

generator = AttentionGenerator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

criterion = nn.BCELoss()

optimizer_G = optim.Adam(
    list(generator.parameters()) +
    list(text_embedding.parameters()),
    lr=LR,
    betas=(0.5, 0.999)
)

optimizer_D = optim.Adam(
    discriminator.parameters(),
    lr=LR,
    betas=(0.5, 0.999)
)


# ==========================================
# Training
# ==========================================

for epoch in range(EPOCHS):

    for real_images, labels in dataloader:

        real_images = real_images.to(DEVICE)
        labels = labels.to(DEVICE)

        batch_size = real_images.size(0)

        real_targets = torch.ones(
            batch_size,
            1,
            device=DEVICE
        )

        fake_targets = torch.zeros(
            batch_size,
            1,
            device=DEVICE
        )

        # ----------------------------------
        # Train Discriminator
        # ----------------------------------

        optimizer_D.zero_grad()

        real_output = discriminator(
            real_images,
            labels
        )

        real_loss = criterion(
            real_output,
            real_targets
        )

        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            device=DEVICE
        )

        fake_labels = torch.randint(
            0,
            NUM_CLASSES,
            (batch_size,),
            device=DEVICE
        )

        fake_images = generator(
            noise,
            fake_labels
        )

        fake_output = discriminator(
            fake_images.detach(),
            fake_labels
        )

        fake_loss = criterion(
            fake_output,
            fake_targets
        )

        discriminator_loss = (
            real_loss + fake_loss
        ) / 2

        discriminator_loss.backward()

        optimizer_D.step()

        # ----------------------------------
        # Train Generator
        # ----------------------------------

        optimizer_G.zero_grad()

        noise = torch.randn(
            batch_size,
            LATENT_DIM,
            device=DEVICE
        )

        generated_labels = torch.randint(
            0,
            NUM_CLASSES,
            (batch_size,)
        ).to(DEVICE)

        generated_images = generator(
            noise,
            generated_labels
        )

        output = discriminator(
            generated_images,
            generated_labels
        )

        generator_loss = criterion(
            output,
            real_targets
        )

        generator_loss.backward()

        optimizer_G.step()

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Generator Loss: {generator_loss.item():.4f} "
        f"Discriminator Loss: {discriminator_loss.item():.4f}"
    )

    # ----------------------------------
    # Generate one image for each text
    # description
    # ----------------------------------

    sample_noise = torch.randn(
        NUM_CLASSES,
        LATENT_DIM,
        device=DEVICE
    )

    sample_labels = torch.arange(
        NUM_CLASSES,
        device=DEVICE
    )

    with torch.no_grad():

        samples = generator(
            sample_noise,
            sample_labels
        )

    save_image(
        samples,
        f"{OUTPUT_DIR}/epoch_{epoch + 1}.png",
        nrow=5,
        normalize=True
    )


print()
print("TASK 5 ATTENTION-BASED GAN COMPLETED!")
print()
print("Text conditions used:")

for i, text in enumerate(text_descriptions):
    print(f"{i}: {text}")

print()
print("Generated images saved to:")
print(OUTPUT_DIR)