#!/usr/bin/env python3
"""[1][2][4] 데이터 샘플링 + 윈도우 단위 데이터셋 구성.

dev 발화 집합에 대해, 해상도 R의 세그먼트 라벨과 1:1로 정렬된 그룹별 풀링 특징,
윈도우 라벨, 그리고 원본 발화 인덱스(발화 단위 분할/탐지 점수용)를 만든다.

라벨 규약 뒤집기: 정답 프레임 '0'(가짜) -> y=1 (양성=가짜).
"""
import os, hashlib
import numpy as np
import soundfile as sf

import ps_data as P
import features as F

CACHE = os.path.join(P.REPO, "cache")


def utt_type(arr):
    s = set(arr.tolist())
    return "bonafide" if s == {"1"} else ("full" if s == {"0"} else "partial")


def categorize(uids, R):
    seg = P.load_seglab(R)
    out = {"bonafide": [], "partial": [], "full": []}
    for u in uids:
        out[utt_type(seg[u])].append(u)
    return out["bonafide"], out["partial"], out["full"]


def sample_uids(R, n_bona=500, n_partial=750, n_full=250, seed=0):
    with open(os.path.join(P.DB, "dev", "dev.lst")) as f:
        all_ids = [l.strip() for l in f if l.strip()]
    bona, partial, full = categorize(all_ids, R)
    rng = np.random.RandomState(seed)

    def pick(pool, k):
        pool = sorted(pool)
        if k >= len(pool):
            return pool
        idx = sorted(rng.choice(len(pool), size=k, replace=False))
        return [pool[i] for i in idx]

    return pick(bona, n_bona) + pick(partial, n_partial) + pick(full, n_full)


def build(uids, R=0.16, groups=F.GROUPS, stats=F.POOL_STATS, cache=True):
    """-> (dict{그룹: X}, y, 발화그룹, uids)."""
    key = hashlib.md5(("v2|" + "|".join(uids) + f"|{R}|{','.join(groups)}|"
                       f"{','.join(stats)}").encode()).hexdigest()[:12]
    path = os.path.join(CACHE, f"win_{R}_{key}.npz")
    if cache and os.path.exists(path):
        d = np.load(path, allow_pickle=True)
        X = {k[2:]: d[k] for k in d.files if k.startswith("X_")}
        return X, d["y"], d["groups"], list(d["uids"])

    seg = P.load_seglab(R)
    Xs = {g: [] for g in groups}
    ys, gidx = [], []
    for gi, u in enumerate(uids):
        a, _ = sf.read(os.path.join(P.WAV, u + ".wav"))
        if a.ndim > 1:
            a = a[:, 0]
        labs = seg[u]
        n = len(labs)
        feats = F.group_features(a.astype(np.float32), n, R=R,
                                 groups=groups, stats=stats)
        m = min(n, *(v.shape[0] for v in feats.values()))
        for g in groups:
            Xs[g].append(feats[g][:m])
        ys.append((np.asarray(labs[:m]) == "0").astype(np.int64))
        gidx.append(np.full(m, gi, dtype=np.int64))

    X = {g: np.concatenate(v) for g, v in Xs.items()}
    y = np.concatenate(ys); gidx = np.concatenate(gidx)
    if cache:
        os.makedirs(CACHE, exist_ok=True)
        np.savez_compressed(path, y=y, groups=gidx, uids=np.array(uids),
                            **{f"X_{g}": v for g, v in X.items()})
    return X, y, gidx, uids


if __name__ == "__main__":
    R = 0.16
    uids = sample_uids(R, 60, 90, 30, seed=0)
    X, y, g, _ = build(uids, R=R, cache=False)
    print(f"utts={len(uids)} windows={len(y)} spoof={y.mean():.3f}")
    for k, v in X.items():
        print(f"  {k:6s} {v.shape}")
