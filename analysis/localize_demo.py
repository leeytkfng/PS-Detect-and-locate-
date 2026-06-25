#!/usr/bin/env python3
"""Visualize spoof *localization* on a held-out partial-spoof utterance.

Trains the best feature (STFT log-mag, configurable) on the train split, then
for one test utterance overlays:
  - the spectrogram,
  - ground-truth spoof spans (red shaded),
  - the model's per-window P(spoof) with its decision threshold.

Run:  python3 src/localize_demo.py [feature] [uid]
"""
import os, sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit

import ps_data as P
import dataset as D
import features as F
from experiment import eer


def main(feat="stft", uid=None, R=0.16, seed=0):
    uids = D.sample_uids(R, 500, 750, 250, seed=seed)
    X, y, groups, _ = D.build(uids, R=R, cache=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(y)), y, groups))

    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X[feat][tr], y[tr])
    thr = eer(y[te], clf.predict_proba(X[feat][te])[:, 1])[1]

    # pick a test partial-spoof utterance if not given
    test_groups = np.unique(groups[te])
    seg = P.load_seglab(R)
    if uid is None:
        for g in test_groups:
            u = uids[g]
            s = set(seg[u].tolist())
            if len(s) > 1:                       # partial spoof
                uid = u
                break
    print(f"feature={feat}  uid={uid}  threshold={thr:.3f}")

    audio, sr = sf.read(os.path.join(P.WAV, uid + ".wav"))
    if audio.ndim > 1:
        audio = audio[:, 0]
    dur = len(audio) / sr
    labs = seg[uid]
    n = len(labs)
    prob = clf.predict_proba(F.extract_all(audio.astype(np.float32), n, R=R)[feat])[:, 1]
    gt_spoof = (np.asarray(labs) == "0").astype(int)
    t = (np.arange(n) + 0.5) * R

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.specgram(audio, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
    ax1.set_ylabel("freq (Hz)")
    ax1.set_title(f"{uid}  -  spoof localization ({feat}, R={R}s)   red = GT spoof")
    for s, e, lab in P.frames_to_intervals(labs.tolist(), R):
        if lab == "spoof":
            ax1.axvspan(s, min(e, dur), color="red", alpha=0.25)

    ax2.step(t, prob, where="mid", color="C0", label="P(spoof) predicted")
    ax2.fill_between(t, 0, gt_spoof, step="mid", color="red", alpha=0.2,
                     label="GT spoof (1=spoof)")
    ax2.axhline(thr, color="k", ls="--", lw=1, label=f"threshold={thr:.2f}")
    ax2.set_ylim(-0.02, 1.02); ax2.set_xlim(0, dur)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("prob")
    ax2.legend(loc="upper right", fontsize=8)

    figdir = os.path.join(P.REPO, "figures")
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, f"localize_{feat}_{uid}.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print("saved:", out)
    return out


if __name__ == "__main__":
    feat = sys.argv[1] if len(sys.argv) > 1 else "stft"
    uid  = sys.argv[2] if len(sys.argv) > 2 else None
    main(feat, uid)
