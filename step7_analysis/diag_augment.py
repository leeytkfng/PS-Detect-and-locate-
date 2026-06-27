#!/usr/bin/env python3
"""증강이 탐지를 떨어뜨린 이유 분해 (clean vs clean+aug, eval).

  - FAR(진짜->가짜 오탐) vs FRR(가짜 놓침)  : 어느 쪽이 나빠졌나
  - 클래스별 점수 분포                       : 진짜/가짜 점수가 어떻게 이동했나
  - 조작 비율별 탐지율                       : 짧은 조작이 특히 더 나빠졌나

실행:  python3 step7_analysis/diag_augment.py
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import numpy as np
import build_full as B
import model as M
import evaluate as E

GROUPS = ("stft", "lfcc", "phase", "disc"); FEAT = list(GROUPS)


def utt_scores(s, g):
    u = np.unique(g)
    return u, np.array([s[g == x].max() for x in u]), np.array([(s[g == x] > -1).any() for x in u])


def main():
    Xc, yc, gc, _ = B.load_full("train", groups=GROUPS)
    Xa, ya, _, _ = B.load_full("train", groups=GROUPS, augment=True)
    Xe, ye, ge, ue = B.load_full("eval", groups=GROUPS)
    Ac = M.compose({k: Xc[k] for k in FEAT}, FEAT)
    Aa = M.compose({k: Xa[k] for k in FEAT}, FEAT)
    Ae = M.compose({k: Xe[k] for k in FEAT}, FEAT)

    # 발화별 가짜비율 + 점수
    uu = np.unique(ge)
    ratio = np.array([ye[ge == x].mean() for x in uu])
    bona = ratio == 0

    def fit_score(X, y, tag):
        clf = M.make_clf("lgbm"); clf.fit(X, y)
        s = clf.predict_proba(Ae)[:, 1]
        us = np.array([s[ge == x].max() for x in uu])
        return us

    sc = fit_score(Ac, yc, "clean")
    sa = fit_score(np.vstack([Ac, Aa]), np.concatenate([yc, ya]), "aug")

    # EER 임계값(각 모델 자기 기준)
    yt = (~bona).astype(int)
    eer_c, th_c = E.eer(yt, sc); eer_a, th_a = E.eer(yt, sa)
    print(f"# 증강 진단 (eval, 발화 {len(uu)}: 진짜 {bona.sum()} / 가짜 {(~bona).sum()})\n")
    print(f"  Utt-EER:  clean {eer_c:.2f}%  ->  aug {eer_a:.2f}%\n")

    # 공통 동작점(clean의 EER 임계값)에서 FAR/FRR 비교
    for tag, s, th in [("clean", sc, th_c), ("aug(clean th)", sa, th_c)]:
        far = (s[bona] >= th_c).mean()             # 진짜를 가짜로
        frr = (s[~bona] < th_c).mean()             # 가짜를 놓침
        print(f"  [{tag:<13}] FAR(진짜오탐)={far:.3f}  FRR(가짜놓침)={frr:.3f}")

    # 클래스별 점수 평균
    print(f"\n  점수평균  진짜: clean {sc[bona].mean():.3f} -> aug {sa[bona].mean():.3f}"
          f"  | 가짜: clean {sc[~bona].mean():.3f} -> aug {sa[~bona].mean():.3f}")

    # 조작 비율별 탐지율(clean th_c 기준)
    print(f"\n  {'가짜비율':<10}{'수':>6}{'탐지율 clean':>13}{'탐지율 aug':>11}")
    for lo, hi in [(0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, .8), (.8, 1.01)]:
        m = (~bona) & (ratio > lo) & (ratio <= hi)
        if m.sum() < 5:
            continue
        dc = (sc[m] >= th_c).mean(); da = (sa[m] >= th_c).mean()
        print(f"  {f'{int(lo*100)}-{int(hi*100)}%':<10}{m.sum():>6}{dc:>13.2%}{da:>11.2%}")
    print("\n  FAR↑면 '진짜에 입힌 아티팩트로 오탐', FRR↑면 '가짜 단서가 흐려져 놓침'.")


if __name__ == "__main__":
    main()
