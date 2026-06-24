#!/usr/bin/env python3
"""Partial-spoof detection & localization experiment.

For each DSP feature family (LFCC / LFCC+d+dd / STFT) we train a simple
logistic-regression classifier on window-level features and evaluate:

  Localization (window/frame level)
    EER, ROC-AUC, F1, balanced accuracy  for spoof-vs-bonafide windows.

  Detection (utterance level)
    utterance spoof score = max over its windows of P(spoof);
    EER / AUC vs. "does the utterance contain any spoof".

Train/test are split *by utterance* (GroupShuffleSplit) so no utterance leaks
across the split. Run:  python3 src/experiment.py
"""
import os, sys, argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score, roc_curve

import ps_data as P
import dataset as D

FEATURES = ["lfcc", "lfcc_dd", "stft"]
PRETTY = {"lfcc": "LFCC", "lfcc_dd": "LFCC+d+dd", "stft": "STFT-spec"}


def eer(y_true, scores):
    """Equal Error Rate (%) and its threshold."""
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2 * 100, thr[i]


def run(R=0.16, n_bona=500, n_partial=750, n_full=250, seed=0):
    print(f"# PartialSpoof dev | resolution={R}s | seed={seed}")
    uids = D.sample_uids(R, n_bona, n_partial, n_full, seed=seed)
    X, y, groups, _ = D.build(uids, R=R, cache=True)
    bona, partial, full = D.categorize(uids, R)
    print(f"  utts={len(uids)} (bona={len(bona)} partial={len(partial)} full={len(full)})"
          f" | windows={len(y)} | spoof-window rate={y.mean():.3f}\n")

    # utterance-aware split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(y)), y, groups))
    g_te = groups[te]
    # utterance-level ground truth on the test set
    utt_gt_full = np.array([1 if (y[groups == g].any()) else 0
                            for g in range(groups.max() + 1)])

    rows = []
    win_pred = {}
    for feat in FEATURES:
        Xf = X[feat]
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0))
        clf.fit(Xf[tr], y[tr])
        s = clf.predict_proba(Xf[te])[:, 1]          # P(spoof) per test window
        win_pred[feat] = s

        # ---- localization (window level) ----
        w_eer, w_thr = eer(y[te], s)
        w_auc = roc_auc_score(y[te], s)
        w_f1  = f1_score(y[te], (s >= w_thr).astype(int))
        w_bac = balanced_accuracy_score(y[te], (s >= w_thr).astype(int))

        # ---- detection (utterance level) ----
        test_utts = np.unique(g_te)
        u_score = np.array([s[g_te == g].max() for g in test_utts])
        u_true  = utt_gt_full[test_utts]
        u_eer, _ = eer(u_true, u_score)
        u_auc = roc_auc_score(u_true, u_score)

        rows.append((PRETTY[feat], Xf.shape[1], w_eer, w_auc, w_f1, w_bac, u_eer, u_auc))

    # ---- results table ----
    print("## Results")
    hdr = f"{'feature':<11} {'dim':>4} | {'win-EER%':>8} {'win-AUC':>7} "\
          f"{'win-F1':>6} {'win-bAcc':>8} | {'utt-EER%':>8} {'utt-AUC':>7}"
    print(hdr); print("-" * len(hdr))
    for nm, dim, we, wa, wf, wb, ue, ua in rows:
        print(f"{nm:<11} {dim:>4} | {we:>8.2f} {wa:>7.3f} {wf:>6.3f} "
              f"{wb:>8.3f} | {ue:>8.2f} {ua:>7.3f}")
    print("\n(win-* = localization @ window level; utt-* = detection @ utterance level)")
    return rows, win_pred, (X, y, groups, te, uids, R)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bona", type=int, default=500)
    ap.add_argument("--partial", type=int, default=750)
    ap.add_argument("--full", type=int, default=250)
    a = ap.parse_args()
    run(R=a.R, n_bona=a.bona, n_partial=a.partial, n_full=a.full, seed=a.seed)
