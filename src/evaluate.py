#!/usr/bin/env python3
"""[7] Evaluation metrics.

  eer(y, scores)        -> Equal Error Rate (%) + threshold
  window_metrics(...)   -> localization: EER / AUC / F1 / balanced-acc
  utt_metrics(...)      -> detection: utterance EER / AUC (score = max pooling)

range_eer(...) is a placeholder for the official PartialSpoof Range-EER
(metric/RangeEER.py, pyannote-based); to be wired once scores are exported in
the score_ali format.
"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, balanced_accuracy_score


def eer(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2 * 100, thr[i]


def window_metrics(y_true, scores):
    e, t = eer(y_true, scores)
    pred = (scores >= t).astype(int)
    return {"eer": e, "auc": roc_auc_score(y_true, scores),
            "f1": f1_score(y_true, pred),
            "bacc": balanced_accuracy_score(y_true, pred), "thr": t}


def utt_scores(win_scores, utt_groups):
    """detection score per utterance = max window P(spoof)."""
    uids = np.unique(utt_groups)
    return uids, np.array([win_scores[utt_groups == u].max() for u in uids])


def utt_metrics(win_scores, utt_groups, utt_is_spoof):
    uids, us = utt_scores(win_scores, utt_groups)
    yt = utt_is_spoof[uids]
    e, _ = eer(yt, us)
    return {"eer": e, "auc": roc_auc_score(yt, us)}


def range_eer(*a, **k):
    raise NotImplementedError(
        "Use metric/RangeEER.py with exported score_ali pkl (next step).")
