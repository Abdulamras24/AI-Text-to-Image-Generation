# Task 4 – Public Dataset Analysis

## Overview

This task analyzes a public image dataset using Python and provides statistical and visual insights into the dataset.

For this task, the **Oxford-102 Flowers Dataset** was selected. The dataset contains images belonging to 102 different flower categories. Python, PyTorch, Torchvision, NumPy, Pandas, and Matplotlib were used for dataset loading, analysis, visualization, and result generation.

## Objectives

The main objectives of this task were to:

* Load and examine a public image dataset.
* Determine the number of classes and images.
* Analyze the number of images available per class.
* Analyze image resolutions.
* Visualize sample images.
* Create a class-distribution graph.
* Explore the use of class labels as text descriptions/prompts.
* Export the analyzed statistics for further use.

## Dataset

**Dataset:** Oxford-102 Flowers

**Training images analyzed:** 1,020

**Number of classes:** 102

The training split contains 10 images for each of the 102 classes.

> Note: Oxford-102 provides flower class labels rather than natural-language image captions. Therefore, class labels were used as text labels/prompts for the text-related analysis instead of treating them as human-written captions.

## Results

### Dataset Statistics

| Metric                   |    Result |
| ------------------------ | --------: |
| Training Images          |     1,020 |
| Number of Classes        |       102 |
| Minimum Images per Class |        10 |
| Maximum Images per Class |        10 |
| Average Images per Class |        10 |
| Minimum Image Width      |    500 px |
| Maximum Image Width      |    919 px |
| Average Image Width      | 624.49 px |
| Minimum Image Height     |    500 px |
| Maximum Image Height     |    993 px |
| Average Image Height     | 537.78 px |

## Generated Outputs

The Python program generates the following files:

### 1. `class_distribution.png`

A bar chart showing the number of training images available for each flower class.

### 2. `sample_flowers.png`

A visualization containing nine sample images from the dataset with their corresponding class labels.

### 3. `dataset_summary.csv`

A CSV file containing the main dataset statistics, including the number of classes, image counts, and image-resolution information.

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* PyTorch
* Torchvision
* SciPy
* Oxford-102 Flowers Dataset

## Project Structure

```text
Task_4/
│
├── dataset/
│   └── Oxford-102 Flowers Dataset
│
├── task4_dataset_analysis.py
├── class_distribution.png
├── sample_flowers.png
├── dataset_summary.csv
└── README.md
```

## How to Run

### 1. Install the required libraries

```bash
pip install numpy pandas matplotlib pillow torch torchvision scipy
```

### 2. Open the Task 4 directory

```bash
cd Task_4
```

### 3. Run the Python program

```bash
python task4_dataset_analysis.py
```

The program automatically downloads the dataset if it is not already available and performs the complete analysis.

## Conclusion

The analysis successfully examined the Oxford-102 Flowers dataset and identified its class distribution, image count, and resolution characteristics. Sample images and statistical visualizations were also generated. The analysis provides a useful understanding of the dataset before using it for further computer vision or text-to-image related tasks.
