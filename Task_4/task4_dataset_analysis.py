# Task 4 - Public Dataset Analysis
# Oxford-102 Flowers Dataset

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from torchvision.datasets import Flowers102


# ==========================================
# 1. Load Dataset
# ==========================================

print("Task 4 - Dataset Analysis")
print("=" * 40)

print("\nLoading Oxford-102 Flowers dataset...")

dataset = Flowers102(
    root="dataset",
    split="train",
    download=True
)

print("Dataset loaded successfully!")
print("Number of training images:", len(dataset))


# ==========================================
# 2. Basic Dataset Statistics
# ==========================================

labels = [label for _, label in dataset]

number_of_classes = len(set(labels))
class_counts = Counter(labels)

print("\nDataset Statistics")
print("-" * 30)

print("Number of classes:", number_of_classes)

print("\nImages per class:")
print("Minimum:", min(class_counts.values()))
print("Maximum:", max(class_counts.values()))
print("Average:", round(np.mean(list(class_counts.values())), 2))


# ==========================================
# 3. Image Resolution Analysis
# ==========================================

print("\nAnalyzing image resolutions...")

widths = []
heights = []

for image, label in dataset:
    width, height = image.size

    widths.append(width)
    heights.append(height)

print("Minimum width:", min(widths))
print("Maximum width:", max(widths))
print("Average width:", round(np.mean(widths), 2))

print("Minimum height:", min(heights))
print("Maximum height:", max(heights))
print("Average height:", round(np.mean(heights), 2))


# ==========================================
# 4. Class Distribution Graph
# ==========================================

print("\nCreating class distribution graph...")

class_numbers = list(class_counts.keys())
image_counts = list(class_counts.values())

plt.figure(figsize=(15, 6))

plt.bar(class_numbers, image_counts)

plt.xlabel("Flower Class")
plt.ylabel("Number of Images")
plt.title("Oxford-102 Flowers - Class Distribution")

plt.tight_layout()

plt.savefig("class_distribution.png", dpi=300)

plt.show()

print("Class distribution graph saved as class_distribution.png")


# ==========================================
# 5. Display Sample Images
# ==========================================

print("\nDisplaying sample images...")

plt.figure(figsize=(12, 8))

for i in range(9):

    image, label = dataset[i]

    plt.subplot(3, 3, i + 1)

    plt.imshow(image)

    plt.title(f"Flower Class {label}")

    plt.axis("off")

plt.tight_layout()

plt.savefig("sample_flowers.png", dpi=300)

plt.show()

print("Sample images saved as sample_flowers.png")


# ==========================================
# 6. Text Label + Image Analysis
# ==========================================

print("\nText Label Analysis")
print("-" * 30)

print("Oxford-102 does not contain natural-language captions.")
print("Therefore, flower class labels are used as text descriptions/prompts.")

# Example text prompts corresponding to flower classes

sample_text_labels = [
    "A photo of a flower",
    "A photo of a beautiful flower",
    "A photo of a flower belonging to Oxford-102 class"
]

for text in sample_text_labels:
    print("Text description:", text)
    print("Word count:", len(text.split()))
    print()


# ==========================================
# 7. Create Dataset Summary
# ==========================================

summary = {
    "Dataset": "Oxford-102 Flowers",
    "Training Images": len(dataset),
    "Number of Classes": number_of_classes,
    "Images Per Class - Minimum": min(class_counts.values()),
    "Images Per Class - Maximum": max(class_counts.values()),
    "Images Per Class - Average": round(
        np.mean(list(class_counts.values())), 2
    ),
    "Minimum Image Width": min(widths),
    "Maximum Image Width": max(widths),
    "Average Image Width": round(np.mean(widths), 2),
    "Minimum Image Height": min(heights),
    "Maximum Image Height": max(heights),
    "Average Image Height": round(np.mean(heights), 2)
}

summary_df = pd.DataFrame(
    list(summary.items()),
    columns=["Metric", "Value"]
)

summary_df.to_csv(
    "dataset_summary.csv",
    index=False
)

print("Dataset summary saved as dataset_summary.csv")


# ==========================================
# 8. Final Message
# ==========================================

print("\n" + "=" * 40)
print("TASK 4 ANALYSIS COMPLETED SUCCESSFULLY!")
print("=" * 40)

print("\nGenerated files:")
print("1. class_distribution.png")
print("2. sample_flowers.png")
print("3. dataset_summary.csv")