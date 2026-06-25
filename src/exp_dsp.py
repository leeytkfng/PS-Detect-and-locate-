#!/usr/bin/env python3
"""Does targeted DSP (phase + discontinuity) beat / add to STFT magnitude?

Same window-level logistic-regression protocol as experiment.py, comparing:
  stft     : STFT log-magnitude (previous best baseline)
  phase    : per-band IF-deviation + phase flux
  disc     : per-band spectral flux + log-energy derivatives
  stft_pd  : STFT ++ phase ++ disc   (the candidate DSP model front-end)

Run:  python3 src/exp_dsp.py
"""
import os, hashlib
import numpy as np
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, balanced_accuracy_score

import ps_data as P
import dataset as D
import features as F
import features_dsp as FD
from experiment import eer

FEATS = ["stft", "phase", "disc", "stft_pd"]
CACHE = os.path.join(P.REPO, "cache")


def build(uids, R=0.16):
    key = hashlib.md5(("dsp|" + "|".join(uids) + f"|{R}").encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"dsp_{R}_{key}.npz")
    if os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return ({k[2:]: d[k] for k in d.files if k.startswith("X_")},
                d["y"], d["groups"])
    seg = P.load_seglab(R)
    Xs = {k: [] for k in FEATS}
    ys, groups = [], []
    for gi, u in enumerate(uids):
        a, _ = sf.read(os.path.join(P.WAV, u + ".wav"))
        if a.ndim > 1:
            a = a[:, 0]
        a = a.astype(np.float32)
        labs = seg[u]
        n = len(labs)
        base = F.extract_all(a, n, R=R)          # has 'stft'
        dsp = FD.extract_dsp(a, n, R=R)          # phase/disc/stft_pd
        feats = {"stft": base["stft"], "phase": dsp["phase"],
                 "disc": dsp["disc"], "stft_pd": dsp["stft_pd"]}
        m = min(n, *(v.shape[0] for v in feats.values()))
        for k in FEATS:
            Xs[k].append(feats[k][:m])
        ys.append((np.asarray(labs[:m]) == "0").astype(np.int64))
        groups.append(np.full(m, gi, dtype=np.int64))
    X = {k: np.concatenate(v) for k, v in Xs.items()}
    y = np.concatenate(ys); groups = np.concatenate(groups)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(path, y=y, groups=groups,
                        **{f"X_{k}": v for k, v in X.items()})
    return X, y, groups


def run(R=0.16, seed=0):
    uids = D.sample_uids(R, 500, 750, 250, seed=seed)
    X, y, groups = build(uids, R=R)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(y)), y, groups))
    g_te = groups[te]
    utt_gt = np.array([1 if y[groups == g].any() else 0 for g in range(groups.max() + 1)])

    print(f"# DSP feature comparison | R={R}s seed={seed} | windows={len(y)} "
          f"spoof={y.mean():.3f}\n")
    hdr = f"{'feature':<9} {'dim':>4} | {'win-EER%':>8} {'win-AUC':>7} {'win-F1':>6} | {'utt-EER%':>8} {'utt-AUC':>7}"
    print(hdr); print("-" * len(hdr))
    for f in FEATS:
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000, class_weight="balanced"))
        clf.fit(X[f][tr], y[tr])
        s = clf.predict_proba(X[f][te])[:, 1]
        we, wt = eer(y[te], s)
        wa = roc_auc_score(y[te], s); wf = f1_score(y[te], (s >= wt).astype(int))
        tu = np.unique(g_te)
        us = np.array([s[g_te == g].max() for g in tu]); ut = utt_gt[tu]
        ue, _ = eer(ut, us); ua = roc_auc_score(ut, us)
        print(f"{f:<9} {X[f].shape[1]:>4} | {we:>8.2f} {wa:>7.3f} {wf:>6.3f} | {ue:>8.2f} {ua:>7.3f}")


if __name__ == "__main__":
    run()
