#!/usr/bin/env python3
"""PartialSpoof DSP 파이프라인 전체 실행  [1]->[7].

[1] dev 발화 로드               (pipeline.sample_uids)
[2][3][4] 프레임 -> DSP 특징 -> 윈도우 풀링   (pipeline.build / features)
[5][6] 경량 분류기 -> 윈도우 P(가짜) -> max 풀링 탐지   (model)
[7] median 스무딩 + 평가(EER)   (model / evaluate)

특징 조합을 비교하고 국소화(윈도우)·탐지(발화) EER을 후처리 전/후로 출력한다.

실행:  python3 src/run.py [--backend logreg|lgbm] [--R 0.16] [--smooth 5]
"""
import argparse
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import ps_data as P
import pipeline as PL
import model as M
import evaluate as E


def run(R=0.16, seed=0, backend="logreg", smooth=5,
        n_bona=500, n_partial=750, n_full=250):
    uids = PL.sample_uids(R, n_bona, n_partial, n_full, seed=seed)
    X, y, groups, _ = PL.build(uids, R=R)               # 그룹별 특징
    bona, part, full = PL.categorize(uids, R)
    print(f"# PartialSpoof DSP pipeline | R={R}s backend={backend} smooth={smooth}")
    print(f"  utts={len(uids)} (bona={len(bona)} partial={len(part)} full={len(full)})"
          f"  windows={len(y)} spoof={y.mean():.3f}\n")

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(y)), y, groups))
    g_te = groups[te]
    utt_spoof = np.zeros(groups.max() + 1, int)
    for g in range(groups.max() + 1):
        utt_spoof[g] = int(y[groups == g].any())

    # Range-EER용 정답: 더 고운 0.02s 라벨로 spoof 구간(ref)·길이(dur) 1회 준비
    seg_fine = P.load_seglab(0.02)
    test_utts = np.unique(g_te)
    ref_dur = {}
    for g in test_utts:
        u = uids[g]
        ref = [(a, b) for a, b, lab in P.frames_to_intervals(seg_fine[u].tolist(), 0.02)
               if lab == "spoof"]
        n_win = int((g_te == g).sum())
        ref_dur[g] = (ref, n_win * R)

    hdr = (f"{'feature set':<17}{'dim':>5} | {'win-EER':>7} | "
           f"{'Utt-EER':>7}{'Utt-AUC':>8} | {'Range-EER':>9}")
    print(hdr); print("-" * len(hdr))
    for name, grps in M.FEATURE_SETS.items():
        Xtr = M.compose({g: X[g][tr] for g in grps}, grps)
        Xte = M.compose({g: X[g][te] for g in grps}, grps)
        s, _ = M.fit_predict(Xtr, y[tr], Xte, backend=backend)
        wm = E.window_metrics(y[te], s)
        um = E.utt_metrics(s, g_te, utt_spoof)            # Utterance EER (탐지)
        s_sm = M.median_smooth(s, g_te, k=smooth)         # 후처리
        per = [dict(scores=s_sm[g_te == g], dur=ref_dur[g][1], ref=ref_dur[g][0])
               for g in test_utts]
        reer, _ = E.range_eer(per, R)                     # Range-EER (국소화)
        print(f"{name:<17}{Xtr.shape[1]:>5} | {wm['eer']:>7.2f} | "
              f"{um['eer']:>7.2f}{um['auc']:>8.3f} | {reer:>9.2f}")
    print("\nUtt-EER = Utterance EER(탐지) | Range-EER = 공식 구간기반 EER(국소화, "
          "0.02s 정답·median 평활 적용) | win-EER = 내부 참고치")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--R", type=float, default=0.16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--backend", default="logreg")
    ap.add_argument("--smooth", type=int, default=5)
    a = ap.parse_args()
    run(R=a.R, seed=a.seed, backend=a.backend, smooth=a.smooth)
