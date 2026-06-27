#!/usr/bin/env python3
"""[7] 평가 지표.

  eer(y, scores)        -> 동일오류율 EER(%) + 임계값
  window_metrics(...)   -> 국소화: EER / AUC / F1 / 균형정확도
  utt_metrics(...)      -> 탐지: 발화 EER / AUC (점수 = max 풀링)

range_eer(...)는 공식 PartialSpoof Range-EER(metric/RangeEER.py, pyannote 기반)
자리표시자. score_ali 포맷으로 점수를 내보낸 뒤 연동 예정.
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
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


def _spoof_runs_to_annotation(scores, R, th):
    """윈도우 P(가짜) >= th 인 연속 구간을 spoof Segment로 (run-length)."""
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    sp = scores >= th
    i, n = 0, len(sp)
    while i < n:
        if sp[i]:
            j = i
            while j < n and sp[j]:
                j += 1
            ann[Segment(i * R, j * R)] = "spoof"
            i = j
        else:
            i += 1
    return ann


def _intervals_to_annotation(intervals):
    """[(start,end), ...] (정답 spoof 구간) -> pyannote Annotation."""
    from pyannote.core import Annotation, Segment
    ann = Annotation()
    for s, e in intervals:
        ann[Segment(s, e)] = "spoof"
    return ann


def range_eer(per_utt, R, n_th=60):
    """공식 PartialSpoof Range-based EER (Zhang et al. 2023) 재현.

    pyannote DetectionCostFunction(시간/구간 기반)을 임계값 스윕으로 FPR=FNR
    지점을 찾는다. metric/RangeEER.py와 동일 라이브러리·로직이되, 입력만
    그들의 model score_ali pkl 대신 우리 윈도우 점수.

    per_utt : [dict(scores=np[win], dur=float, ref=[(s,e),...]), ...]
    R       : 윈도우(=가설 프레임) 길이[초].  반환: (EER%, threshold)
    """
    from pyannote.metrics.detection import DetectionCostFunction
    from pyannote.core import Segment

    items, allsco = [], []
    for u in per_utt:
        items.append((np.asarray(u["scores"], float),
                      _intervals_to_annotation(u["ref"]),
                      Segment(0, u["dur"])))
        allsco.append(np.asarray(u["scores"], float))
    ths = np.quantile(np.concatenate(allsco), np.linspace(0.02, 0.98, n_th))

    fprs, fnrs = [], []
    for th in ths:
        dcf = DetectionCostFunction()
        for sco, ref, uem in items:
            dcf(reference=ref, hypothesis=_spoof_runs_to_annotation(sco, R, th),
                uem=uem, detailed=False)
        a = dcf.accumulated_
        fpr = a["false alarm"] / a["negative class total"] if a["negative class total"] else 0.0
        fnr = a["miss"] / a["positive class total"] if a["positive class total"] else 0.0
        fprs.append(fpr); fnrs.append(fnr)
    fprs, fnrs = np.array(fprs), np.array(fnrs)
    i = int(np.argmin(np.abs(fprs - fnrs)))
    return (fprs[i] + fnrs[i]) / 2 * 100, float(ths[i])
