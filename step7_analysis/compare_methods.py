#!/usr/bin/env python3
"""tabular 분류기(method) 비교 — 동일 DSP 특징(full)으로 정식 train->dev 평가.

"왜 LightGBM인가"를 데이터로 뒷받침: 같은 특징/프로토콜에서 분류기만 바꿔
Utt-EER(탐지) / Range-EER(국소화) / 학습시간 비교.

  logreg  선형 (베이스라인)
  mlp     얕은 신경망 (tabular)
  rf      랜덤포레스트 (배깅 트리)
  histgb  HistGradientBoosting (sklearn 부스팅)
  lgbm    LightGBM (우리 method)

실행:  python3 src/compare_methods.py
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import time
import numpy as np

import ps_data as P
import build_full as B
import model as M
import evaluate as E
from run_full import utt_is_spoof, reer_subset, _aslist

GROUPS = ("stft", "lfcc", "phase", "disc")
FEAT = ["stft", "lfcc", "phase", "disc"]          # = "full"
BACKENDS = ["logreg", "mlp", "rf", "histgb", "lgbm"]


def main(R=0.16, reer_n=1500, smooth=5):
    Xtr, ytr, gtr, _ = B.load_full("train", R=R, groups=GROUPS)
    Xde, yde, gde, uids = B.load_full("dev", R=R, groups=GROUPS)
    Atr = M.compose({g: Xtr[g] for g in FEAT}, FEAT)
    Ade = M.compose({g: Xde[g] for g in FEAT}, FEAT)
    utt_sp = utt_is_spoof(yde, gde)

    sub, seg002 = reer_subset(uids, gde, reer_n, R, "dev")
    ref_dur = {g: ([(a, b) for a, b, l in
                    P.frames_to_intervals(_aslist(seg002[uids[g]]), 0.02) if l == "spoof"],
                   int((gde == g).sum()) * R) for g in sub}

    print(f"# method 비교 | 특징=full({Atr.shape[1]}d) | train->dev | "
          f"train {len(ytr)}win / dev {len(yde)}win\n")
    print(f"  {'method':<9}{'win-EER':>8}{'Utt-EER':>8}{'Utt-AUC':>8}"
          f"{'Range-EER':>10}{'train_s':>9}")
    print("  " + "-" * 52)
    for b in BACKENDS:
        t0 = time.time()
        try:
            s, _ = M.fit_predict(Atr, ytr, Ade, backend=b)
        except Exception as e:
            print(f"  {b:<9} 실패: {e}")
            continue
        dt = time.time() - t0
        wm = E.window_metrics(yde, s)
        um = E.utt_metrics(s, gde, utt_sp)
        s_sm = M.median_smooth(s, gde, k=smooth)
        per = [dict(scores=s_sm[gde == g], dur=ref_dur[g][1], ref=ref_dur[g][0])
               for g in sub]
        reer, _ = E.range_eer(per, R)
        print(f"  {b:<9}{wm['eer']:>8.2f}{um['eer']:>8.2f}{um['auc']:>8.3f}"
              f"{reer:>10.2f}{dt:>9.1f}")
    print(f"\n  Range-EER = dev 부분집합 {len(sub)}발화. 동일 특징(full), 분류기만 교체.")


if __name__ == "__main__":
    main()
