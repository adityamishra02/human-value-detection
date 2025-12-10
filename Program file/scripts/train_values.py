import argparse, os, random
import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from values_dataset import ValuesMLDataset
from model_pl_values import AdamSmithValues

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def choose_device_string():
    if torch.cuda.is_available():
        return 'cuda'
    try:
        if torch.backends.mps.is_available():
            return 'mps'
    except Exception:
        pass
    return 'cpu'

def main(args):
    seed_everything(args.seed)

    ds_train = ValuesMLDataset(args.sentences, args.labels,
                               tokenizer_name=args.model_name,
                               max_length=args.max_len)
    ds_val = ValuesMLDataset(args.sentences_val, args.labels_val,
                             tokenizer_name=args.model_name,
                             max_length=args.max_len)

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True,
                          num_workers=8, pin_memory=False)
    dl_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                        num_workers=8, pin_memory=False)

    model = AdamSmithValues(model_name=args.model_name, lr=args.lr)

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_cb = ModelCheckpoint(dirpath=args.out_dir,
                              filename='best-{epoch:02d}-{val_sub1_f1:.4f}',
                              monitor='val_sub1_f1', mode='max', save_top_k=1)
    early = EarlyStopping(monitor='val_sub1_f1', patience=3, mode='max')

    device = choose_device_string()
    if device == 'cuda':
        precision = 16
    elif device == 'mps':
        precision = "16-mps"
    else:
        precision = 32

    trainer = Trainer(
        accelerator=None if device == 'cpu' else device,
        devices=1 if device != 'cpu' else None,
        precision=32,  
        max_epochs=args.epochs,
        callbacks=[ckpt_cb, early],
        deterministic=True,
        log_every_n_steps=20

    )


    trainer.fit(model, dl_train, dl_val)
    print("Best model saved to:", ckpt_cb.best_model_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--sentences_val', required=True)
    parser.add_argument('--labels_val', required=True)
    parser.add_argument('--model_name', default='microsoft/deberta-large')
    parser.add_argument('--out_dir', default='./models/run1')
    parser.add_argument('--epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--max_len', type=int, default=165)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
