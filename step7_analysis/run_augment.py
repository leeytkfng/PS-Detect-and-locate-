#!/usr/bin/env python3
"""데이터 증강 효과: clean train vs (clean + 증강) train -> dev/eval 평가.

train만 증강(라벨/타이밍 보존). dev/eval은 깨끗하게 평가. 국소화는 dev에서 고른
gaussian-1.0 평활 적용. 가설: unseen 갭이 '조금' 줄지만 크게는 안 줄 것.

실행:  python3 step7_analysis/run_augment.py
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import numpy as np
from scipy.ndimage import gaussian_filter1d

import ps_data as P
import build_full as B
import model as M
import evaluate as E
from run_full import utt_is_spoof, reer_subset, _aslist

GROUPS = ("stft", "lfcc", "phase", "disc"); FEAT = list(GROUPS); R = 0.16


def gsmooth(s, g, sigma=1.0):
    out = s.copy()
    for u in np.unique(g):
        idx = np.where(g == u)[0]
        out[idx] = gaussian_filter1d(s[idx], sigma=sigma, mode="nearest")
    return out


def prep(split, uids, groups, n=1500):
    sub, seg = reer_subset(uids, groups, n, R, split)
    rd = {x: ([(a, b) for a, b, l in P.frames_to_intervals(_aslist(seg[uids[x]]), 0.02)
              if l == "spoof"], int((groups == x).sum()) * R) for x in sub}
    return sub, rd


def evaluate_on(clf, split):
    X, y, g, u = B.load_full(split, groups=GROUPS)
    s = clf.predict_proba(M.compose({k: X[k] for k in FEAT}, FEAT))[:, 1]
    um = E.utt_metrics(s, g, utt_is_spoof(y, g))
    sub, rd = prep(split, u, g)
    per = [dict(scores=gsmooth(s, g)[g == x], dur=rd[x][1], ref=rd[x][0]) for x in sub]
    return um["eer"], E.range_eer(per, R)[0]


def main():
    Xc, yc, gc, _ = B.load_full("train", groups=GROUPS)
    Xa, ya, ga, _ = B.load_full("train", groups=GROUPS, augment=True)
    Ac = M.compose({g: Xc[g] for g in FEAT}, FEAT)
    Aa = M.compose({g: Xa[g] for g in FEAT}, FEAT)

    print(f"# 증강 효과 (LightGBM full, gaussian-1.0 평활) | train clean {len(yc)} / +aug {len(yc)+len(ya)} win\n")
    print(f"  {'train':<14}{'dev Utt':>8}{'dev Range':>10}{'eval Utt':>9}{'eval Range':>11}")
    # clean
    clf = M.make_clf("lgbm"); clf.fit(Ac, yc)
    du, dr = evaluate_on(clf, "dev"); eu, er = evaluate_on(clf, "eval")
    print(f"  {'clean':<14}{du:>8.2f}{dr:>10.2f}{eu:>9.2f}{er:>11.2f}")
    # clean + aug
    clf = M.make_clf("lgbm"); clf.fit(np.vstack([Ac, Aa]), np.concatenate([yc, ya]))
    du, dr = evaluate_on(clf, "dev"); eu, er = evaluate_on(clf, "eval")
    print(f"  {'clean+aug':<14}{du:>8.2f}{dr:>10.2f}{eu:>9.2f}{er:>11.2f}")
    print("\n  train만 증강(noise/reverb/codec/rawboost). dev/eval 깨끗. unseen 갭 변화 관찰.")


if __name__ == "__main__":
    main()
