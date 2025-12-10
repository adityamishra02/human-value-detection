#!/usr/bin/env python3
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
    set_seed
)
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

# -----------------------------
# Config (paper setup)
# -----------------------------
MODEL_NAME = "FacebookAI/roberta-large"
NUM_LABELS = 38   
BATCH_SIZE = 8
LR = 2e-5
EPOCHS = 4
WARMUP_RATIO = 0.2
SEED = 42
UPSAMPLE_FACTOR = 4   # paper used ×4
set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# Paper's chosen upsample categories (14)
# -----------------------------
PAPER_UPSAMPLE_COLS = [
    "Self-direction: thought constrained",
    "Self-direction: action constrained",
    "Humility attained",
    "Humility constrained",
    "Face attained",
    "Face constrained",
    "Benevolence: caring constrained",
    "Benevolence: dependability constrained",
    "Universalism: tolerance attained",
    "Universalism: tolerance constrained",
    "Conformity: interpersonal attained",
    "Conformity: interpersonal constrained",
    "Tradition constrained",
    "Power: dominance constrained"
]


# -----------------------------
# Dataset
# -----------------------------
class ValuesDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len=256):
        self.df = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        text = str(self.df.iloc[idx]["text"])
        labels = self.df.iloc[idx]["labels"]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(labels, dtype=torch.float)
        }


# -----------------------------
# F1 Loss (paper used F1-loss)
# -----------------------------
class F1Loss(nn.Module):
    def __init__(self, epsilon=1e-7):
        super(F1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)
        tp = (y_true * y_pred).sum(dim=0)
        fp = ((1 - y_true) * y_pred).sum(dim=0)
        fn = (y_true * (1 - y_pred)).sum(dim=0)
        f1 = (2 * tp + self.epsilon) / (2 * tp + fp + fn + self.epsilon)
        return 1 - f1.mean()


# -----------------------------
# Upsampling (only paper’s 14 categories)
# -----------------------------
def upsample_paper_categories(df, factor=UPSAMPLE_FACTOR):
    upsampled_rows = [df]
    for col in PAPER_UPSAMPLE_COLS:
        if col not in df.columns:
            continue
        subset = df[df[col] == 1]
        if not subset.empty:
            upsampled_rows.append(pd.concat([subset] * (factor - 1), ignore_index=True))
    return pd.concat(upsampled_rows, ignore_index=True)


# -----------------------------
# Training Loop
# -----------------------------
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion):
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask).logits
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        val_f1 = evaluate_model(model, val_loader)

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "best_roberta_model.pt")

    print("Best Validation F1:", best_f1)


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, data_loader):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu().numpy()

            outputs = model(input_ids, attention_mask=attention_mask).logits
            preds_batch = torch.sigmoid(outputs).cpu().numpy()
            preds.append(preds_batch)
            true_labels.append(labels)

    preds = np.vstack(preds)
    true_labels = np.vstack(true_labels)
    preds_binary = (preds > 0.5).astype(int)

    return f1_score(true_labels, preds_binary, average="macro")


# -----------------------------
# Data Loading (TSV merge)
# -----------------------------
def load_dataset(sentences_path, labels_path):
    sentences = pd.read_csv(sentences_path, sep="\t")
    labels = pd.read_csv(labels_path, sep="\t")

    df = pd.merge(sentences, labels, on=["Text-ID", "Sentence-ID"])
    label_cols = df.columns.difference(["Text-ID", "Sentence-ID", "Text"])
    df["labels"] = df[label_cols].values.tolist()
    df = df.rename(columns={"Text": "text"})
    return df[["text", "labels"] + list(label_cols)]  


# -----------------------------
# Main
# -----------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # File names (English TSVs)
    train_df = load_dataset("sentences_training.tsv", "labels_training.tsv")
    val_df   = load_dataset("sentences_validation.tsv", "labels_validation.tsv")
    test_df  = load_dataset("sentences_test.tsv", "labels_test.tsv")

    # Upsample train set (paper’s 14 categories)
    train_df = upsample_paper_categories(train_df)

    # Dataset & Dataloader
    train_ds = ValuesDataset(train_df, tokenizer)
    val_ds = ValuesDataset(val_df, tokenizer)
    test_ds = ValuesDataset(test_df, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        problem_type="multi_label_classification"
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(WARMUP_RATIO * total_steps)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    criterion = F1Loss()

    # Train
    train_model(model, train_loader, val_loader, optimizer, scheduler, criterion)

    # Test
    model.load_state_dict(torch.load("best_roberta_model.pt"))
    test_f1 = evaluate_model(model, test_loader)
    print("Test Macro F1:", test_f1)


if __name__ == "__main__":
    main()
