#!/usr/bin/env python3
"""국소화(Range-EER) 후처리 실험: 점수 시퀀스 평활(median/gaussian).

Range-EER은 임계값 스윕 기반이라 '점수를 매끄럽게' 하는 후처리만 EER로 측정 가능
(min-duration·hysteresis는 고정 동작점 도구라 별도). dev에서 최적 평활을 고르고
그대로 eval에 적용한다(테스트 누수 방지).

실행:  python3 step7_analysis/postproc_localize.py
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

import ps_data as P
import build_full as B
import model as M
import evaluate as E
from run_full import utt_is_spoof, reer_subset, _aslist

GROUPS = ("stft", "lfcc", "phase", "disc")
FEAT = list(GROUPS)


def smooth(scores, groups, kind, param):
    if kind == "none":
        return scores
    out = scores.copy()
    for u in np.unique(groups):
        idx = np.where(groups == u)[0]
        s = scores[idx]
        if kind == "median":
            out[idx] = median_filter(s, size=param, mode="nearest")
        else:                       # gaussian
            out[idx] = gaussian_filter1d(s, sigma=param, mode="nearest")
    return out


def reer_of(scores, groups, sub, ref_dur, R):
    per = [dict(scores=scores[groups == g], dur=ref_dur[g][1], ref=ref_dur[g][0])
           for g in sub]
    return E.range_eer(per, R)[0]


def prep(split, uids, groups, R, n=1200):
    sub, seg = reer_subset(uids, groups, n, R, split)
    ref_dur = {g: ([(a, b) for a, b, l in P.frames_to_intervals(_aslist(seg[uids[g]]), 0.02)
                    if l == "spoof"], int((groups == g).sum()) * R) for g in sub}
    return sub, ref_dur


def main(R=0.16):
    Xtr, ytr, gtr, _ = B.load_full("train", groups=GROUPS)
    Xde, yde, gde, ude = B.load_full("dev", groups=GROUPS)
    Xev, yev, gev, uev = B.load_full("eval", groups=GROUPS)
    Atr = M.compose({g: Xtr[g] for g in FEAT}, FEAT)
    print("train lgbm(full) ...")
    clf = M.make_clf("lgbm"); clf.fit(Atr, ytr)
    sde = clf.predict_proba(M.compose({g: Xde[g] for g in FEAT}, FEAT))[:, 1]
    sev = clf.predict_proba(M.compose({g: Xev[g] for g in FEAT}, FEAT))[:, 1]

    sub_d, rd_d = prep("dev", ude, gde, R)
    sub_e, rd_e = prep("eval", uev, gev, R)

    configs = [("none", 0), ("median", 5), ("median", 7), ("median", 9),
               ("median", 11), ("gauss", 1.0), ("gauss", 2.0)]
    print(f"\n# 후처리 평활 tuning (dev에서 선택) | R={R}s")
    print(f"  {'config':<14}{'dev Range-EER':>14}")
    best = None
    for kind, p in configs:
        s = smooth(sde, gde, "gaussian" if kind == "gauss" else kind, p)
        r = reer_of(s, gde, sub_d, rd_d, R)
        print(f"  {kind+'-'+str(p):<14}{r:>14.2f}")
        if best is None or r < best[0]:
            best = (r, kind, p)
    _, bk, bp = best
    # eval: baseline(현행 median-5) vs dev-최적
    base_e = reer_of(smooth(sev, gev, "median", 5), gev, sub_e, rd_e, R)
    best_e = reer_of(smooth(sev, gev, "gaussian" if bk == "gauss" else bk, bp),
                     gev, sub_e, rd_e, R)
    print(f"\n# dev 최적 = {bk}-{bp} (dev {best[0]:.2f})")
    print(f"# eval Range-EER:  baseline(median-5) = {base_e:.2f}  ->  best({bk}-{bp}) = {best_e:.2f}")
    print("  (dev에서 고른 평활을 eval에 그대로 적용 — 테스트 누수 없음)")


if __name__ == "__main__":
    main()
