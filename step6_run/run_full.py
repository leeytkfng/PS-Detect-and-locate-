#!/usr/bin/env python3
"""정식 프로토콜 평가: train 으로 학습 -> dev/eval 로 평가 (전체 데이터).

dev 내부분할(낙관적)이 아니라, 공식 train->dev / train->eval 로 신뢰 수치를 낸다.
사전에 build_full.py 로 각 split 특징 캐시를 만들어둬야 함.

  - Utterance EER / window EER : 평가 split 전체
  - Range-EER : 비용이 커서 평가 split의 대표 부분집합(--reer_n)으로 측정

실행:  python3 src/run_full.py --test dev --backend lgbm
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import argparse
import numpy as np

import ps_data as P
import build_full as B
import model as M
import evaluate as E

GROUPS = ("stft", "lfcc", "phase", "disc")          # seam 제외(헤드라인 full엔 불필요)
SETS = {k: v for k, v in M.FEATURE_SETS.items()
        if set(v) <= set(GROUPS)}                    # lfcc/stft/stft+phase+disc/full


def utt_is_spoof(y, groups):
    out = np.zeros(groups.max() + 1, int)
    for g in range(groups.max() + 1):
        out[g] = int(y[groups == g].any())
    return out


def _aslist(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


def reer_subset(uids, groups, n, R, test, seed=0):
    """Range-EER용 대표 부분집합 발화 인덱스(진짜/부분/완전 고루)."""
    seg = B.load_split_seglab(test, 0.02)          # split별 0.02s 정답 (dev 하드코딩 금지)

    def typ(u):
        s = set(_aslist(seg[u]))
        return 0 if s == {"1"} else (2 if s == {"0"} else 1)
    by = {0: [], 1: [], 2: []}
    for g in range(groups.max() + 1):
        by[typ(uids[g])].append(g)
    rng = np.random.RandomState(seed)
    per = max(1, n // 3)
    pick = []
    for t in (0, 1, 2):
        pool = by[t]
        pick += list(rng.choice(pool, min(per, len(pool)), replace=False)) if pool else []
    return sorted(pick), seg


def main(test="dev", backend="lgbm", R=0.16, smooth=5, reer_n=2000):
    print(f"# 정식 프로토콜 | train -> {test} | backend={backend} | R={R}s")
    Xtr, ytr, gtr, _ = B.load_full("train", R=R, groups=GROUPS)
    Xte, yte, gte, uids_te = B.load_full(test, R=R, groups=GROUPS)
    print(f"  train: {gtr.max()+1} utts / {len(ytr)} win (spoof {ytr.mean():.3f})")
    print(f"  {test}: {gte.max()+1} utts / {len(yte)} win (spoof {yte.mean():.3f})\n")

    utt_sp = utt_is_spoof(yte, gte)
    sub, seg002 = reer_subset(uids_te, gte, reer_n, R, test)
    sub_set = set(sub)
    ref_dur = {}
    for g in sub:
        ref = [(a, b) for a, b, l in P.frames_to_intervals(_aslist(seg002[uids_te[g]]), 0.02)
               if l == "spoof"]
        ref_dur[g] = (ref, int((gte == g).sum()) * R)

    hdr = (f"{'feature set':<17}{'dim':>5} | {'win-EER':>7} | "
           f"{'Utt-EER':>7}{'Utt-AUC':>8} | {'Range-EER':>9}")
    print(hdr); print("-" * len(hdr))
    for name, grps in SETS.items():
        Atr = M.compose({g: Xtr[g] for g in grps}, grps)
        Ate = M.compose({g: Xte[g] for g in grps}, grps)
        s, _ = M.fit_predict(Atr, ytr, Ate, backend=backend)
        wm = E.window_metrics(yte, s)
        um = E.utt_metrics(s, gte, utt_sp)
        s_sm = M.median_smooth(s, gte, k=smooth)
        per = [dict(scores=s_sm[gte == g], dur=ref_dur[g][1], ref=ref_dur[g][0])
               for g in sub]
        reer, _ = E.range_eer(per, R)
        print(f"{name:<17}{Atr.shape[1]:>5} | {wm['eer']:>7.2f} | "
              f"{um['eer']:>7.2f}{um['auc']:>8.3f} | {reer:>9.2f}")
    print(f"\nUtt-EER/win-EER = {test} 전체 | Range-EER = {test} 부분집합 {len(sub)}발화 "
          f"(median 평활). 정식 프로토콜(train 학습) 신뢰 수치.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="dev")
    ap.add_argument("--backend", default="lgbm")
    ap.add_argument("--R", type=float, default=0.16)
    ap.add_argument("--smooth", type=int, default=5)
    ap.add_argument("--reer_n", type=int, default=2000)
    a = ap.parse_args()
    main(test=a.test, backend=a.backend, R=a.R, smooth=a.smooth, reer_n=a.reer_n)
