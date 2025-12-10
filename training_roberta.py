

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer, RobertaModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.cuda.amp import autocast, GradScaler

class ValuesDataset(Dataset):
    def __init__(self, df, tokenizer, num_values, max_len=128):
        self.texts = df["text"].astype(str).tolist()
        self.y_val = df[[f"value_{i}" for i in range(num_values)]].values
        self.y_attain = df[[f"attain_{i}" for i in range(num_values)]].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y_val": torch.tensor(self.y_val[idx], dtype=torch.float),
            "y_attain": torch.tensor(self.y_attain[idx], dtype=torch.float),
        }



class RobertaTeacherModel(nn.Module):
    def __init__(self, num_values):
        super().__init__()
        self.encoder = RobertaModel.from_pretrained("roberta-large")
        hidden_size = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.value_head = nn.Linear(hidden_size, num_values)
        self.attain_head = nn.Linear(hidden_size, num_values)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_emb = outputs.last_hidden_state[:, 0, :]  
        cls_emb = self.dropout(cls_emb)
        logits_val = self.value_head(cls_emb)
        logits_attain = self.attain_head(cls_emb)
        return logits_val, logits_attain


def evaluate(model, data_loader, device, threshold=0.5, desc="Validation"):
    model.eval()
    y_true_val, y_pred_val = [], []
    y_true_attain, y_pred_attain = [], []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Evaluating ({desc})", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_val = batch["y_val"].cpu().numpy()
            y_attain = batch["y_attain"].cpu().numpy()

            logits_val, logits_attain = model(input_ids, attention_mask)
            probs_val = torch.sigmoid(logits_val).cpu().numpy()
            probs_attain = torch.sigmoid(logits_attain).cpu().numpy()

            preds_val = (probs_val > threshold).astype(int)
            preds_attain = (probs_attain > threshold).astype(int)

            y_true_val.append(y_val)
            y_pred_val.append(preds_val)
            y_true_attain.append(y_attain)
            y_pred_attain.append(preds_attain)

    y_true_val = np.vstack(y_true_val)
    y_pred_val = np.vstack(y_pred_val)
    y_true_attain = np.vstack(y_true_attain)
    y_pred_attain = np.vstack(y_pred_attain)

    f1_val = f1_score(y_true_val, y_pred_val, average="micro", zero_division=0)
    f1_attain = f1_score(y_true_attain, y_pred_attain, average="micro", zero_division=0)
    acc_val = accuracy_score(y_true_val.argmax(axis=1), y_pred_val.argmax(axis=1))
    acc_attain = accuracy_score(y_true_attain.argmax(axis=1), y_pred_attain.argmax(axis=1))

    return f1_val, f1_attain, acc_val, acc_attain

def train(model, train_loader, val_loader, test_loader, device, num_epochs=3, lr=2e-5, save_dir="checkpoints_roberta"):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            y_val = batch["y_val"].to(device)
            y_attain = batch["y_attain"].to(device)

            with autocast():
                logits_val, logits_attain = model(input_ids, attention_mask)
                loss_val = loss_fn(logits_val, y_val)
                loss_attain = loss_fn(logits_attain, y_attain)
                loss = 0.5 * loss_val + 0.5 * loss_attain

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{total_loss/(pbar.n+1):.4f}"})

        avg_loss = total_loss / len(train_loader)
        f1_val, f1_attain, acc_val, acc_attain = evaluate(model, val_loader, device, desc="Validation")
        print(f"\nEpoch {epoch+1}: Loss={avg_loss:.4f}")
        print(f"Validation → F1(Value)={f1_val:.4f}, F1(Attain)={f1_attain:.4f}, "
              f"Acc(Value)={acc_val:.4f}, Acc(Attain)={acc_attain:.4f}")

        torch.save(model.state_dict(), f"{save_dir}/roberta_teacher_epoch{epoch+1}.pt")

    print("\n🔎 Final Evaluation on Test Set:")
    f1_v, f1_a, acc_v, acc_a = evaluate(model, test_loader, device, desc="Test")
    print(f"Test → F1(Value)={f1_v:.4f}, F1(Attain)={f1_a:.4f}, Acc(Value)={acc_v:.4f}, Acc(Attain)={acc_a:.4f}")

    torch.save(model.state_dict(), f"{save_dir}/roberta_teacher_final.pt")
    print("\n✅ Training complete! Model saved to", save_dir)


if __name__ == "__main__":
    TRAIN_PATH = "Data/final_training_augmented.tsv"
    VAL_PATH = "Data/final_validation_real.tsv"
    TEST_PATH = "Data/final_test_real.tsv"

    df_train = pd.read_csv(TRAIN_PATH, sep="\t")
    df_val = pd.read_csv(VAL_PATH, sep="\t")
    df_test = pd.read_csv(TEST_PATH, sep="\t")

    NUM_VALUES = len([c for c in df_train.columns if c.startswith("value_")])
    print(f" Loaded train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}, num_values={NUM_VALUES}")

    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")

    train_dataset = ValuesDataset(df_train, tokenizer, num_values=NUM_VALUES, max_len=128)
    val_dataset = ValuesDataset(df_val, tokenizer, num_values=NUM_VALUES, max_len=128)
    test_dataset = ValuesDataset(df_test, tokenizer, num_values=NUM_VALUES, max_len=128)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2)
    test_loader = DataLoader(test_dataset, batch_size=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("🖥️ Using device:", device)

    model = RobertaTeacherModel(NUM_VALUES)
    train(model, train_loader, val_loader, test_loader, device, num_epochs=3, lr=1.5e-5)
