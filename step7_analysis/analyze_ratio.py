#!/usr/bin/env python3
"""조작 비율(intra-spoof ratio)별 탐지 난이도 분해 — 부분조작 특화 분석.

가설: 발화 내 가짜 구간 비율이 낮을수록(짧을수록) 탐지가 어렵다.
train(full)으로 LightGBM 학습 -> eval 예측 -> 발화별 탐지점수(max-pool)와
가짜 프레임 비율 계산 -> 비율 구간별로 (해당 구간 spoof vs 전체 bonafide) EER.

실행:  python3 src/analyze_ratio.py
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import numpy as np
import build_full as B
import model as M
import evaluate as E

GROUPS = ("stft", "lfcc", "phase", "disc")
FEAT = list(GROUPS)


def per_utt(scores, y, groups):
    """발화별 (탐지점수=max P(spoof), 가짜비율=mean(y))."""
    u = np.unique(groups)
    sc = np.array([scores[groups == g].max() for g in u])
    ratio = np.array([y[groups == g].mean() for g in u])
    return sc, ratio


def main():
    Xtr, ytr, gtr, _ = B.load_full("train", groups=GROUPS)
    Xte, yte, gte, _ = B.load_full("eval", groups=GROUPS)
    Atr = M.compose({g: Xtr[g] for g in FEAT}, FEAT)
    Ate = M.compose({g: Xte[g] for g in FEAT}, FEAT)
    print("training lgbm(full) on train, predicting eval ...")
    s, _ = M.fit_predict(Atr, ytr, Ate, backend="lgbm")
    sc, ratio = per_utt(s, yte, gte)

    bona = ratio == 0                      # 전체 진짜
    print(f"\n# 조작 비율별 탐지 EER (eval, train->eval, LightGBM full)")
    print(f"  진짜 발화 {bona.sum()} | 가짜 발화 {(~bona).sum()}\n")
    bins = [(0.0, 0.1), (0.1, 0.2), (0.2, 0.3), (0.3, 0.5), (0.5, 0.8), (0.8, 1.01)]
    print(f"  {'가짜비율 구간':<14}{'발화수':>7}{'EER%':>9}{'탐지율@τ':>10}")
    # 전체 EER 임계값(τ) = 전체 spoof vs bona EER 지점
    eer_all, thr = E.eer(np.r_[np.zeros(bona.sum()), np.ones((~bona).sum())],
                         np.r_[sc[bona], sc[~bona]])
    for lo, hi in bins:
        m = (~bona) & (ratio > lo) & (ratio <= hi)
        if m.sum() < 5:
            continue
        yy = np.r_[np.zeros(bona.sum()), np.ones(m.sum())]
        ss = np.r_[sc[bona], sc[m]]
        e, _ = E.eer(yy, ss)
        det = (sc[m] >= thr).mean()        # 고정 임계값에서 탐지율(recall)
        print(f"  {f'{int(lo*100)}-{int(hi*100)}%':<14}{m.sum():>7}{e:>9.2f}{det:>10.2%}")
    print(f"\n  전체 spoof vs bona EER = {eer_all:.2f}%  (τ={thr:.3f})")
    print("  -> 가짜 비율이 낮을수록(짧을수록) EER↑·탐지율↓ 이면 '짧은 조작일수록 어렵다' 입증.")


if __name__ == "__main__":
    main()
