#!/usr/bin/env python3
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, classification_report

# -----------------------------
# Config
# -----------------------------
MODEL_NAME = "aishanur/HVD_Roberta_Large_Upsampled"
BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# Data Loading (TSV merge)
# -----------------------------
def load_dataset(sentences_path, labels_path):
    sentences = pd.read_csv(sentences_path, sep="\t")
    labels = pd.read_csv(labels_path, sep="\t")

    df = pd.merge(sentences, labels, on=["Text-ID", "Sentence-ID"])
    label_cols = df.columns.difference(["Text-ID", "Sentence-ID", "Text"])
    df["labels"] = df[label_cols].values.tolist()
    df = df.rename(columns={"Text": "text"})
    return df[["text", "labels"]], label_cols.tolist()


# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, data_loader, label_names):
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

    # Overall macro-F1
    macro_f1 = f1_score(true_labels, preds_binary, average="macro")

    # Per-class F1
    report = classification_report(
        true_labels,
        preds_binary,
        target_names=label_names,
        zero_division=0,
        output_dict=True
    )

    return macro_f1, report


# -----------------------------
# Main
# -----------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).to(device)

    # Load test set
    test_df, label_names = load_dataset("sentences_test.tsv", "labels_test.tsv")
    test_ds = ValuesDataset(test_df, tokenizer)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    # Evaluate
    macro_f1, report = evaluate_model(model, test_loader, label_names)

    print(f"\nOverall Test Macro F1: {macro_f1:.4f}\n")

    print("Per-label F1 scores:")
    for label in label_names:
        f1 = report[label]["f1-score"]
        support = report[label]["support"]
        print(f"{label:40s} F1={f1:.4f} (n={support})")


if __name__ == "__main__":
    main()
