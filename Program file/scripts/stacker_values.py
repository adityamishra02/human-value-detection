import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
import joblib

def train_stacker(preds_list, y_true, save_path=None):
    """
    preds_list: list of np arrays [(N,L), (N,L), ...] from each base model. Concatenated horizontally.
    y_true: (N,L)
    Returns trained MultiOutputClassifier
    """
    X = np.concatenate(preds_list, axis=1)
    clf = MultiOutputClassifier(LogisticRegression(max_iter=400), n_jobs=-1)
    clf.fit(X, y_true)
    if save_path:
        joblib.dump(clf, save_path)
    return clf

def predict_stacker(clf, preds_list):
    X = np.concatenate(preds_list, axis=1)
    return clf.predict(X)
