import os
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

VALUE_NAMES = [
    "Self-direction: thought", "Self-direction: action",
    "Stimulation", "Hedonism", "Achievement",
    "Power: dominance", "Power: resources",
    "Face",
    "Security: personal", "Security: societal",
    "Tradition",
    "Conformity: rules", "Conformity: interpersonal",
    "Humility",
    "Benevolence: caring", "Benevolence: dependability",
    "Universalism: concern", "Universalism: nature", "Universalism: tolerance"
]

class ValuesMLDataset(Dataset):
    def __init__(self, sentences_tsv, labels_tsv, tokenizer_name='microsoft/deberta-large', task=1, max_length=256):
        """
        task: 1 -> subtask1 (19-dim presence)
              2 -> subtask2 (38-dim attained/constrained)
              0 -> both (returns both labels_sub1 and labels_sub2)
        """
        assert os.path.exists(sentences_tsv), f"Missing {sentences_tsv}"
        assert os.path.exists(labels_tsv), f"Missing {labels_tsv}"
        self.sentences = pd.read_csv(sentences_tsv, sep='\t', dtype=str).fillna('')
        self.labels = pd.read_csv(labels_tsv, sep='\t', dtype=str).fillna('')
        merged = pd.merge(self.sentences, self.labels, on=['Text-ID','Sentence-ID'], how='inner')
        merged = merged.rename(columns={'Text': 'text'})
        self.df = merged.reset_index(drop=True)
        non_label_cols = {'Text-ID','Sentence-ID','text'}
        self.label_cols = [c for c in self.df.columns if c not in non_label_cols]
        if len(self.label_cols) != 38:
            raise ValueError(f"Expected 38 label columns, found {len(self.label_cols)}. Columns: {self.label_cols}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.max_length = max_length
        self.task = task

    def __len__(self):
        return len(self.df)

    def _parse_label_row(self, row):
        vals = []
        for col in self.label_cols:
            cell = row[col].strip()
            if cell == '':
                vals.append(0.0)
            else:
                try:
                    vals.append(float(cell))
                except:
                    vals.append(0.0)
        sub2 = [1 if v == 1.0 or v == 0.5 else 0 for v in vals]
        sub1 = []
        for i in range(0, 38, 2):
            a = vals[i]
            c = vals[i+1]
            present = 1 if (a > 0.0 or c > 0.0) else 0
            sub1.append(present)
        return torch.tensor(sub1, dtype=torch.float), torch.tensor(sub2, dtype=torch.float)

    def __getitem__(self, idx):
        row = self.df.loc[idx]
        text = str(row['text'])
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        labels_sub1, labels_sub2 = self._parse_label_row(row)
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels_sub1'] = labels_sub1
        item['labels_sub2'] = labels_sub2
        item['meta_TextID'] = row['Text-ID']
        item['meta_SentenceID'] = row['Sentence-ID']
        return item
