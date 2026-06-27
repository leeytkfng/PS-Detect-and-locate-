#!/usr/bin/env python3
"""[5][6][7] 분류기 + 탐지 풀링 + 후처리.

윈도우 단위 경량 분류기가 P(가짜)를 예측한다. 백엔드:
  - "logreg" : StandardScaler + LogisticRegression (기본, 빠른 베이스라인)
  - "lgbm"   : LightGBM 설치 시 사용 (더 강력, 선택)

  국소화 = 윈도우별 P(가짜) 시퀀스
  탐지   = 발화 내 윈도우들의 최댓값
  후처리 = 발화별 점수 시퀀스에 median(중앙값) 스무딩
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# 대표 특징 조합 (pipeline 그룹들의 조합)
FEATURE_SETS = {
    "lfcc":            ["lfcc"],
    "stft":            ["stft"],
    "stft+phase+disc": ["stft", "phase", "disc"],
    "full":            ["stft", "lfcc", "phase", "disc"],
}


def compose(Xdict, names):
    return np.concatenate([Xdict[n] for n in names], axis=1)


def make_clf(backend="logreg"):
    """tabular 분류기 백엔드. 윈도우 단위 특징은 정형(tabular) 데이터라 트리/부스팅이 강함.
    윈도우 라벨은 거의 균형(spoof~0.5)이라 class_weight 영향은 작음."""
    if backend == "lgbm":
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(n_estimators=300, learning_rate=0.05,
                                  num_leaves=63, class_weight="balanced",
                                  subsample=0.8, colsample_bytree=0.8,
                                  verbosity=-1)
        except Exception:
            pass  # 실패하면 조용히 logreg로 폴백
    if backend == "xgb":                                 # XGBoost 부스팅
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                             subsample=0.8, colsample_bytree=0.8,
                             tree_method="hist", n_jobs=-1, eval_metric="logloss")
    if backend == "rf":                                  # 배깅 트리
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                      min_samples_leaf=5, max_features="sqrt",
                                      class_weight="balanced")
    if backend == "histgb":                              # sklearn 부스팅(빠름)
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                              max_leaf_nodes=63)
    if backend == "mlp":                                 # 얕은 신경망(tabular)
        from sklearn.neural_network import MLPClassifier
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=60,
                          early_stopping=True))
    return make_pipeline(                                # logreg (선형 베이스라인)
        StandardScaler(),
        LogisticRegression(max_iter=3000, class_weight="balanced"))


def fit_predict(Xtr, ytr, Xte, backend="logreg"):
    clf = make_clf(backend)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1], clf


def median_smooth(scores, utt_groups, k=5):
    """발화별 윈도우 점수 시퀀스에 median 필터 적용 (후처리)."""
    if k < 3 or k % 2 == 0:
        return scores
    out = scores.copy()
    half = k // 2
    for u in np.unique(utt_groups):
        idx = np.where(utt_groups == u)[0]
        s = scores[idx]
        sm = np.array([np.median(s[max(0, i - half):i + half + 1])
                       for i in range(len(s))])
        out[idx] = sm
    return out
