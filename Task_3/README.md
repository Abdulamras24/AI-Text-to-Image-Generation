# Task 3 - Text Preprocessing using Hugging Face Transformers

## Objective

The objective of this task is to build a text preprocessing pipeline using Hugging Face Transformers. The system converts natural-language descriptions into tokenized and numerical representations that can be used by downstream AI systems.

## Technologies Used

* Python
* Hugging Face Transformers
* DistilBERT
* PyTorch
* Pandas

## Model Used

**DistilBERT (`distilbert-base-uncased`)**

DistilBERT is a lightweight Transformer-based language model. It was used to tokenize the input text and generate 768-dimensional text representations.

## Input Text Descriptions

The following example descriptions were used:

1. A beautiful red flower
2. A bright yellow sunflower
3. A pink rose in a garden
4. A purple flower with green leaves
5. A white flower in nature

## Processing Pipeline

```text
Text Description
       ↓
Hugging Face Tokenizer
       ↓
Tokens
       ↓
Input IDs + Attention Mask
       ↓
DistilBERT
       ↓
Text Embeddings
       ↓
CSV Output
```

## Tokenization

The tokenizer converts text into smaller tokens and maps them to numerical IDs.

For example:

```text
A bright yellow sunflower
        ↓
a bright yellow sun ##flower
```

The `##flower` token is a subword continuation of `sunflower`.

## Encoded Representation

Each text description is converted into:

* Input IDs
* Attention Mask

The Input IDs represent the tokens numerically, while the Attention Mask identifies which positions contain meaningful tokens.

## Text Embeddings

DistilBERT was used to generate hidden representations for each text description.

Each input produced an embedding with the shape:

```text
[1, 768]
```

Therefore, every text description was represented using a 768-dimensional numerical vector.

These embeddings provide semantic representations that can be used by downstream AI applications.

## Output Files

### `task3_text_preprocessing.py`

Python program containing the complete preprocessing and embedding-generation pipeline.

### `text_preprocessing_results.csv`

Contains:

* Original text
* Tokens
* Token count
* Input IDs
* Attention Mask

### `text_embeddings.csv`

Contains:

* Original text
* 768 embedding values for each text description

## Technical Note

DistilBERT embeddings were used in this task to demonstrate text preprocessing and semantic representation. These embeddings are not directly compatible with every text-to-image architecture. A production text-to-image system would normally use the text encoder specified by the target model, such as a CLIP-based encoder.

## Result

The task was successfully completed. The system can convert text descriptions into tokenized representations, numerical encodings, and 768-dimensional text embeddings using Hugging Face Transformers.
