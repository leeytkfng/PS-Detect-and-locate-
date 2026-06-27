#!/usr/bin/env python3
"""전체 데이터(train/dev/eval) 윈도우 특징을 멀티프로세싱으로 추출·캐시.

split별 con_wav/<split>.lst/segment_labels(<split>_seglab_R.npy)를 읽어
그룹별 풀링 특징 + 윈도우 라벨 + 발화인덱스를 만들어 cache/full_<split>_*.npz 로 저장.
직렬이면 eval 7만 발화에 수 시간 -> Pool 로 코어 수만큼 가속.

실행:  python3 src/build_full.py dev            # dev 전체
       python3 src/build_full.py train 0.16     # train, 해상도 0.16
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import os, sys, hashlib
import numpy as np
import soundfile as sf
from multiprocessing import Pool

import ps_data as P
import features as F

CACHE = os.path.join(P.REPO, "cache")
N_JOBS = max(1, min(16, (os.cpu_count() or 4) - 2))


def split_paths(split):
    wav = os.path.join(P.DB, split, "con_wav")
    lst = os.path.join(P.DB, split, f"{split}.lst")
    return wav, lst


def load_split_seglab(split, R):
    return np.load(os.path.join(P.SEG, f"{split}_seglab_{R:.2f}.npy"),
                   allow_pickle=True).item()


# 워커: (uid, 라벨배열, wavdir, R, groups) -> (uid, {g:X}, y)  (없으면 None)
def _work(task):
    uid, labs, wavdir, R, groups = task
    try:
        a, _ = sf.read(os.path.join(wavdir, uid + ".wav"))
    except Exception:
        return None
    if a.ndim > 1:
        a = a[:, 0]
    n = len(labs)
    feats = F.group_features(a.astype(np.float32), n, R=R, groups=groups)
    m = min(n, *(v.shape[0] for v in feats.values()))
    y = (np.asarray(labs[:m]) == "0").astype(np.int64)
    return uid, {g: feats[g][:m] for g in groups}, y


def build_split(split, R=0.16, groups=F.GROUPS, n_jobs=N_JOBS, limit=None):
    key = hashlib.md5(f"{split}|{R}|{','.join(groups)}|{limit}".encode()).hexdigest()[:10]
    out = os.path.join(CACHE, f"full_{split}_{R}_{key}.npz")
    if os.path.exists(out):
        print("cached:", out); return out

    wavdir, lstpath = split_paths(split)
    seg = load_split_seglab(split, R)
    with open(lstpath) as f:
        uids = [l.strip() for l in f if l.strip()]
    if limit:
        uids = uids[:limit]
    tasks = [(u, seg[u], wavdir, R, groups) for u in uids if u in seg]
    print(f"{split}: {len(tasks)} utts, {n_jobs} jobs, groups={groups}")

    Xs = {g: [] for g in groups}
    ys, gidx, kept = [], [], []
    with Pool(n_jobs) as pool:
        for i, res in enumerate(pool.imap(_work, tasks, chunksize=16)):
            if res is None:
                continue
            uid, feats, y = res
            gi = len(kept); kept.append(uid)
            for g in groups:
                Xs[g].append(feats[g])
            ys.append(y); gidx.append(np.full(len(y), gi, np.int64))
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{len(tasks)}")

    X = {g: np.concatenate(v) for g, v in Xs.items()}
    y = np.concatenate(ys); gidx = np.concatenate(gidx)
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(out, y=y, groups=gidx, uids=np.array(kept),
                        **{f"X_{g}": v for g, v in X.items()})
    print(f"saved: {out}  | utts={len(kept)} windows={len(y)} spoof={y.mean():.3f}")
    return out


def load_full(split, R=0.16, groups=F.GROUPS, limit=None):
    key = hashlib.md5(f"{split}|{R}|{','.join(groups)}|{limit}".encode()).hexdigest()[:10]
    d = np.load(os.path.join(CACHE, f"full_{split}_{R}_{key}.npz"), allow_pickle=True)
    X = {k[2:]: d[k] for k in d.files if k.startswith("X_")}
    return X, d["y"], d["groups"], list(d["uids"])


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "dev"
    R = float(sys.argv[2]) if len(sys.argv) > 2 else 0.16
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    build_split(split, R=R, limit=lim)
