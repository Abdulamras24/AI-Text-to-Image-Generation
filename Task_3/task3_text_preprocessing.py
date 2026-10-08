# Task 3 - Text Preprocessing using Hugging Face Transformers

from transformers import AutoTokenizer, AutoModel
import pandas as pd
import torch


# ==========================================
# 1. Load Tokenizer
# ==========================================

print("Task 3 - Text Preprocessing")
print("=" * 40)

print("\nLoading Hugging Face tokenizer...")

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

print("Tokenizer loaded successfully!")
print("\nLoading DistilBERT model...")
model = AutoModel.from_pretrained("distilbert-base-uncased")

print("DistilBERT model loaded successfully!")

model.eval()

# ==========================================
# 2. Define Text Descriptions
# ==========================================

texts = [
    "A beautiful red flower",
    "A bright yellow sunflower",
    "A pink rose in a garden",
    "A purple flower with green leaves",
    "A white flower in nature"
]

print("\nNumber of text descriptions:", len(texts))


# ==========================================
# 3. Process Each Text
# ==========================================

results = []

for text in texts:

    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Encode the text
    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)

    token_embeddings = outputs.last_hidden_state

    attention_mask = encoded["attention_mask"].unsqueeze(-1)

    masked_embeddings = token_embeddings * attention_mask

    sum_embeddings = masked_embeddings.sum(dim=1)

    sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)

    embedding = sum_embeddings / sum_mask

    embedding_values = embedding.numpy()[0]

    print("\nText:", text)
    print("Embedding shape:", embedding.shape)
    # Store the results
    results.append({
        "Text": text,
        "Tokens": " ".join(tokens),
        "Token Count": len(tokens),
        "Input IDs": encoded["input_ids"].tolist()[0],
        "Attention Mask": encoded["attention_mask"].tolist()[0]
    })

# ==========================================
# 5. Generate and Save Text Embeddings
# ==========================================

print("\nGenerating text embeddings...")

embedding_results = []

for text in texts:

    encoded = tokenizer(
        text,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**encoded)

    token_embeddings = outputs.last_hidden_state

    attention_mask = encoded["attention_mask"].unsqueeze(-1)

    masked_embeddings = token_embeddings * attention_mask

    sum_embeddings = masked_embeddings.sum(dim=1)

    sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)

    embedding = sum_embeddings / sum_mask

    embedding_values = embedding.numpy()[0]

    embedding_results.append(
        [text] + embedding_values.tolist()
    )


# Create embedding column names

embedding_columns = ["Text"]

for i in range(768):
    embedding_columns.append(f"Embedding_{i+1}")


embedding_df = pd.DataFrame(
    embedding_results,
    columns=embedding_columns
)


embedding_df.to_csv(
    "text_embeddings.csv",
    index=False
)

print("Text embeddings saved as text_embeddings.csv")
# ==========================================
# 4. Display Results
# ==========================================

print("\nText Processing Results")
print("-" * 40)

for result in results:

    print("\nText:", result["Text"])
    print("Tokens:", result["Tokens"])
    print("Token Count:", result["Token Count"])
    print("Input IDs:", result["Input IDs"])
    print("Attention Mask:", result["Attention Mask"])


# ==========================================
# 5. Create DataFrame
# ==========================================

df = pd.DataFrame(results)

print("\n\nDataFrame:")
print(df)


# ==========================================
# 6. Save Results
# ==========================================

df.to_csv(
    "text_preprocessing_results.csv",
    index=False
)

print("\nResults saved as text_preprocessing_results.csv")


# ==========================================
# 7. Final Message
# ==========================================

print("\n" + "=" * 40)
print("TASK 3 TEXT PREPROCESSING COMPLETED!")
print("=" * 40)