#!/usr/bin/env python3
"""[7] 평가 지표.

  eer(y, scores)        -> 동일오류율 EER(%) + 임계값
  window_metrics(...)   -> 국소화: EER / AUC / F1 / 균형정확도
  utt_metrics(...)      -> 탐지: 발화 EER / AUC (점수 = max 풀링)

range_eer(...)는 공식 PartialSpoof Range-EER(metric/RangeEER.py, pyannote 기반)
자리표시자. score_ali 포맷으로 점수를 내보낸 뒤 연동 예정.
"""
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, balanced_accuracy_score


def eer(y_true, scores):
    fpr, tpr, thr = roc_curve(y_true, scores)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return (fpr[i] + fnr[i]) / 2 * 100, thr[i]


def window_metrics(y_true, scores):
    e, t = eer(y_true, scores)
    pred = (scores >= t).astype(int)
    return {"eer": e, "auc": roc_auc_score(y_true, scores),
            "f1": f1_score(y_true, pred),
            "bacc": balanced_accuracy_score(y_true, pred), "thr": t}


def utt_scores(win_scores, utt_groups):
    """발화별 탐지 점수 = 윈도우 P(가짜)의 최댓값."""
    uids = np.unique(utt_groups)
    return uids, np.array([win_scores[utt_groups == u].max() for u in uids])


def utt_metrics(win_scores, utt_groups, utt_is_spoof):
    uids, us = utt_scores(win_scores, utt_groups)
    yt = utt_is_spoof[uids]
    e, _ = eer(yt, us)
    return {"eer": e, "auc": roc_auc_score(yt, us)}


def range_eer(*a, **k):
    raise NotImplementedError(
        "Use metric/RangeEER.py with exported score_ali pkl (next step).")
