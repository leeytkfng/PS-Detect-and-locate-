#!/usr/bin/env python3
"""End-to-end PartialSpoof DSP pipeline  [1]->[7].

[1] load dev utterances        (pipeline.sample_uids)
[2][3][4] frame -> DSP features -> window pooling   (pipeline.build / features)
[5][6] light classifier -> window P(spoof) -> detection by max pool   (model)
[7] median smoothing + evaluation (EER)            (model / evaluate)

Compares feature sets and reports localization (window) + detection (utt) EER,
with and without post-processing.

Run:  python3 src/run.py [--backend logreg|lgbm] [--R 0.16] [--smooth 5]
"""
import argparse
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import pipeline as PL
import model as M
import evaluate as E


def run(R=0.16, seed=0, backend="logreg", smooth=5,
        n_bona=500, n_partial=750, n_full=250):
    uids = PL.sample_uids(R, n_bona, n_partial, n_full, seed=seed)
    X, y, groups, _ = PL.build(uids, R=R)               # per-group features
    bona, part, full = PL.categorize(uids, R)
    print(f"# PartialSpoof DSP pipeline | R={R}s backend={backend} smooth={smooth}")
    print(f"  utts={len(uids)} (bona={len(bona)} partial={len(part)} full={len(full)})"
          f"  windows={len(y)} spoof={y.mean():.3f}\n")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(y)), y, groups))
    g_te = groups[te]
    utt_spoof = np.zeros(groups.max() + 1, int)
    for g in range(groups.max() + 1):
        utt_spoof[g] = int(y[groups == g].any())

    hdr = (f"{'feature set':<17}{'dim':>5} | {'win-EER':>7}{'win-AUC':>8}"
           f"{'win-F1':>7} | {'utt-EER':>7}{'utt-AUC':>8} | {'win-EER+sm':>10}")
    print(hdr); print("-" * len(hdr))
    for name, grps in M.FEATURE_SETS.items():
        Xtr = M.compose({g: X[g][tr] for g in grps}, grps)
        Xte = M.compose({g: X[g][te] for g in grps}, grps)
        s, _ = M.fit_predict(Xtr, y[tr], Xte, backend=backend)
        wm = E.window_metrics(y[te], s)
        um = E.utt_metrics(s, g_te, utt_spoof)
        s_sm = M.median_smooth(s, g_te, k=smooth)
        wm_sm = E.window_metrics(y[te], s_sm)
        print(f"{name:<17}{Xtr.shape[1]:>5} | {wm['eer']:>7.2f}{wm['auc']:>8.3f}"
              f"{wm['f1']:>7.3f} | {um['eer']:>7.2f}{um['auc']:>8.3f} | "
              f"{wm_sm['eer']:>10.2f}")
    print("\nwin-* localization (per window) | utt-* detection (max pool) | "
          "+sm = median-smoothed window EER")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="logreg")
    ap.add_argument("--smooth", type=int, default=5)
    a = ap.parse_args()
    run(R=a.R, seed=a.seed, backend=a.backend, smooth=a.smooth)
