#!/usr/bin/env python3
"""Build window-level feature matrices + labels for a set of dev utterances.

Each utterance is split into segment-label windows (resolution R). For every
window we get:
  - feature vectors for the 3 families (lfcc / lfcc_dd / stft)
  - a binary label  (1 = spoof window, 0 = bonafide window)   [note the flip!]
  - the originating utterance index (for utterance-aware splitting / scoring)

We flip the dataset convention to the usual "positive = spoof" so EER / AUC
read naturally:  ground-truth frame '0' (spoof) -> y=1,  '1' (bonafide) -> y=0.
"""
import os, hashlib
import numpy as np
import soundfile as sf

import ps_data as P
import features as F

CACHE = os.path.join(P.REPO, "cache")


def categorize(uids, R):
    """Split ids into bonafide / partial-spoof / full-spoof by segment labels."""
    seg = P.load_seglab(R)
    bona, partial, full = [], [], []
    for u in uids:
        s = set(seg[u].tolist())
        if s == {"1"}:
            bona.append(u)
        elif s == {"0"}:
            full.append(u)
        else:
            partial.append(u)
    return bona, partial, full


def sample_uids(R, n_bona=500, n_partial=750, n_full=250, seed=0):
    """Deterministic balanced-ish sample across the three utterance types."""
    with open(os.path.join(P.DB, "dev", "dev.lst")) as f:
        all_ids = [l.strip() for l in f if l.strip()]
    bona, partial, full = categorize(all_ids, R)
    rng = np.random.RandomState(seed)

    def pick(pool, k):
        pool = sorted(pool)
        if k >= len(pool):
            return pool
        idx = rng.choice(len(pool), size=k, replace=False)
        return [pool[i] for i in sorted(idx)]

    chosen = pick(bona, n_bona) + pick(partial, n_partial) + pick(full, n_full)
    return chosen


def build(uids, R=0.16, cache=True):
    """-> dict(X by feature name), y (window labels), groups (utt idx), uids."""
    key = hashlib.md5(("|".join(uids) + f"|{R}").encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"ds_{R}_{key}.npz")
    if cache and os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        return ({k[2:]: d[k] for k in d.files if k.startswith("X_")},
                d["y"], d["groups"], list(d["uids"]))

    seg = P.load_seglab(R)
    Xs = {"lfcc": [], "lfcc_dd": [], "stft": []}
    ys, groups = [], []
    for gi, u in enumerate(uids):
        audio, _ = sf.read(os.path.join(P.WAV, u + ".wav"))
        if audio.ndim > 1:
            audio = audio[:, 0]
        labs = seg[u]
        n = len(labs)
        feats = F.extract_all(audio.astype(np.float32), n, R=R)
        m = min(n, *(v.shape[0] for v in feats.values()))   # safety align
        for k in Xs:
            Xs[k].append(feats[k][:m])
        # ground-truth '0'=spoof -> positive class 1
        y = (np.asarray(labs[:m]) == "0").astype(np.int64)
        ys.append(y)
        groups.append(np.full(m, gi, dtype=np.int64))

    X = {k: np.concatenate(v, axis=0) for k, v in Xs.items()}
    y = np.concatenate(ys)
    groups = np.concatenate(groups)

    if cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, y=y, groups=groups, uids=np.array(uids),
                            **{f"X_{k}": v for k, v in X.items()})
    return X, y, groups, uids


if __name__ == "__main__":
    R = 0.16
    uids = sample_uids(R, 60, 90, 30, seed=0)      # tiny smoke test
    X, y, g, _ = build(uids, R=R, cache=False)
    print(f"utts={len(uids)}  windows={len(y)}  "
          f"spoof={y.mean():.3f}  groups={g.max()+1}")
    for k, v in X.items():
        print(f"  {k:8s} {v.shape}")
