#!/usr/bin/env python3
"""[5][6][7] Classifier + detection pooling + post-processing.

A light window-level classifier predicts P(spoof). Backends:
  - "logreg"   : StandardScaler + LogisticRegression (default, fast baseline)
  - "lgbm"     : LightGBM if installed (stronger, optional)

  localization = per-window P(spoof) sequence
  detection    = max over an utterance's windows
  post-process = per-utterance median smoothing of the score sequence
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# canonical feature sets (combinations of pipeline groups)
FEATURE_SETS = {
    "lfcc":            ["lfcc"],
    "stft":            ["stft"],
    "stft+phase+disc": ["stft", "phase", "disc"],
    "full":            ["stft", "lfcc", "phase", "disc"],
}


def compose(Xdict, names):
    return np.concatenate([Xdict[n] for n in names], axis=1)


def make_clf(backend="logreg"):
    if backend == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                  num_leaves=63, class_weight="balanced",
                                  subsample=0.8, colsample_bytree=0.8,
                                  verbosity=-1)
        except Exception:
            pass  # fall back silently
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"))


def fit_predict(Xtr, ytr, Xte, backend="logreg"):
    clf = make_clf(backend)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1], clf


def median_smooth(scores, utt_groups, k=5):
    """per-utterance median filter on the window-score sequence (post-proc)."""
    if k < 3 or k % 2 == 0:
        return scores
    out = scores.copy()
    half = k // 2
    for u in np.unique(utt_groups):
        idx = np.where(utt_groups == u)[0]
        s = scores[idx]
        sm = np.array([np.median(s[max(0, i - half):i + half + 1])
                       for i in range(len(s))])
        out[idx] = sm
    return out
