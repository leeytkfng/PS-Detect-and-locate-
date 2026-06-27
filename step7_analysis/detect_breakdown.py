#!/usr/bin/env python3
"""정직한 탐지 분해: 발화 단위 탐지가 정말 좋은가, 아니면 쉬운 완전가짜
발화 덕에 부풀려진 것인가?

발화 탐지(점수=윈도우 P(가짜)의 max)를 세 경우로 분해:
  - 진짜 vs 전체 가짜      (헤드라인 숫자)
  - 진짜 vs 완전가짜       (쉬운 경우)
  - 진짜 vs 부분가짜       (어렵고 현실적인 경우)

실행:  python3 analysis/detect_breakdown.py
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

import ps_data as P
import pipeline as D
import model as MD
from evaluate import eer


def main(R=0.16, seed=0):
    uids = D.sample_uids(R, 500, 750, 250, seed=seed)
    X, y, groups, _ = D.build(uids, R=R, cache=True)
    seg = P.load_seglab(R)

    # 발화 타입: 0=진짜, 1=부분가짜, 2=완전가짜
    def utype(u):
        s = set(seg[u].tolist())
        return 0 if s == {"1"} else (2 if s == {"0"} else 1)
    typ = np.array([utype(uids[g]) for g in range(groups.max() + 1)])

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(y)), y, groups))
    g_te = groups[te]
    test_utts = np.unique(g_te)

    print(f"# detection breakdown | R={R}s seed={seed}")
    n_b = (typ[test_utts] == 0).sum()
    n_p = (typ[test_utts] == 1).sum()
    n_f = (typ[test_utts] == 2).sum()
    print(f"  test utts: bonafide={n_b} partial={n_p} full={n_f}\n")

    hdr = f"{'feature':<11} | {'bona-vs-ALL':>12} | {'bona-vs-FULL':>13} | {'bona-vs-PARTIAL':>15}"
    sub = f"{'':<11} | {'EER%  AUC':>12} | {'EER%   AUC':>13} | {'EER%    AUC':>15}"
    print(hdr); print(sub); print("-" * len(hdr))

    for feat, grps in MD.FEATURE_SETS.items():
        Xtr = MD.compose({g: X[g][tr] for g in grps}, grps)
        Xte = MD.compose({g: X[g][te] for g in grps}, grps)
        s, _ = MD.fit_predict(Xtr, y[tr], Xte)
        uscore = {g: s[g_te == g].max() for g in test_utts}

        def pair(pos_type):
            ids = [g for g in test_utts if typ[g] in (0, pos_type)]
            yy = np.array([0 if typ[g] == 0 else 1 for g in ids])
            ss = np.array([uscore[g] for g in ids])
            if len(set(yy)) < 2:
                return float("nan"), float("nan")
            return eer(yy, ss)[0], roc_auc_score(yy, ss)

        e_all_ids = list(test_utts)
        y_all = np.array([0 if typ[g] == 0 else 1 for g in e_all_ids])
        s_all = np.array([uscore[g] for g in e_all_ids])
        eA, aA = eer(y_all, s_all)[0], roc_auc_score(y_all, s_all)
        eF, aF = pair(2)
        eP, aP = pair(1)
        print(f"{feat:<11} | {eA:5.1f} {aA:5.3f} | {eF:5.1f} {aF:5.3f}  | "
              f"{eP:5.1f}  {aP:5.3f}")

    print("\nbona-vs-PARTIAL is the realistic threat; bona-vs-FULL is the easy case.")


if __name__ == "__main__":
    main()
