import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from values_dataset import ValuesMLDataset
from model_pl_values import AdamSmithValues
from threshold_search import find_optimal_threshold

def load_checkpoint_flexible(ckpt_path, model_name='microsoft/deberta-large', device='cpu'):
    """Load checkpoint into model, tolerant to lightning prefixes"""
    model = AdamSmithValues(model_name=model_name)
    state = torch.load(ckpt_path, map_location='cpu')
    sd = state.get('state_dict', state)
    new_sd = {}
    for k, v in sd.items():
        newk = k
        if k.startswith('model.'):
            newk = k[len('model.'):]
        if k.startswith('lightning_module.'):
            newk = k[len('lightning_module.'):]
        new_sd[newk] = v
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    model.to(device)
    return model

def predict_probs(model, dataset, batch_size=32, device='cpu'):
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_sub1 = []
    all_sub2 = []
    with torch.no_grad():
        for b in dl:
            ids = b['input_ids'].to(device)
            mask = b['attention_mask'].to(device)
            logits1, logits2 = model(ids, mask)
            probs1 = torch.sigmoid(logits1).cpu().numpy()
            probs2 = torch.sigmoid(logits2).cpu().numpy()
            all_sub1.append(probs1)
            all_sub2.append(probs2)
    return np.vstack(all_sub1), np.vstack(all_sub2)

def average_ensemble(ckpt_paths, leave_sentences, leave_labels, test_sentences, test_labels=None, model_name='microsoft/deberta-large', batch_size=32, device='cpu'):
    """
    ckpt_paths: list of checkpoint file paths
    leave_sentences/leave_labels: files for threshold selection
    test_sentences/test_labels: files for test predictions; test_labels optional (if available)
    Returns: test_pred1, test_pred2, thresholds (t1, t2)
    """
    leave_ds = ValuesMLDataset(leave_sentences, leave_labels, tokenizer_name=model_name)
    test_ds = ValuesMLDataset(test_sentences, test_labels, tokenizer_name=model_name) if test_labels else ValuesMLDataset(test_sentences, leave_labels, tokenizer_name=model_name)

    preds_leave1 = []
    preds_leave2 = []
    preds_test1 = []
    preds_test2 = []

    for ckpt in ckpt_paths:
        print("Loading", ckpt)
        m = load_checkpoint_flexible(ckpt, model_name=model_name, device=device)
        l1, l2 = predict_probs(m, leave_ds, batch_size=batch_size, device=device)
        t1, t2 = predict_probs(m, test_ds, batch_size=batch_size, device=device)
        preds_leave1.append(l1); preds_leave2.append(l2)
        preds_test1.append(t1); preds_test2.append(t2)

    leave1_avg = np.mean(np.stack(preds_leave1, axis=0), axis=0)
    leave2_avg = np.mean(np.stack(preds_leave2, axis=0), axis=0)
    test1_avg = np.mean(np.stack(preds_test1, axis=0), axis=0)
    test2_avg = np.mean(np.stack(preds_test2, axis=0), axis=0)

    y_leave1 = np.vstack([x['labels_sub1'].numpy() for x in leave_ds])
    y_leave2 = np.vstack([x['labels_sub2'].numpy() for x in leave_ds])

    t1, f1 = find_optimal_threshold(y_leave1, leave1_avg)
    t2, f2 = find_optimal_threshold(y_leave2, leave2_avg)
    print(f"Selected thresholds: sub1 {t1} (f1={f1:.4f}), sub2 {t2} (f1={f2:.4f})")

    test_pred1 = (test1_avg >= t1).astype(int)
    test_pred2 = (test2_avg >= t2).astype(int)
    return test_pred1, test_pred2, (t1, t2)
