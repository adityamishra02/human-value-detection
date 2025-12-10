import numpy as np
from sklearn.metrics import precision_score, recall_score

def find_optimal_threshold(y_true, y_probs, step=0.01):
    """
    Single global threshold search (returns best t and best F1).
    y_true: (N, L) binary
    y_probs: (N, L) floats
    """
    assert y_true.shape == y_probs.shape
    thresholds = np.arange(0.0, 1.0+1e-9, step)
    best_t = 0.0
    best_f1 = -1.0
    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        prec = precision_score(y_true, preds, average='macro', zero_division=0)
        rec = recall_score(y_true, preds, average='macro', zero_division=0)
        if (prec + rec) == 0:
            f1 = 0.0
        else:
            f1 = 2*prec*rec/(prec+rec)
        if f1 >= best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1
