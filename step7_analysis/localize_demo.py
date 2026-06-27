#!/usr/bin/env python3
"""홀드아웃 부분가짜 발화에서 가짜 *국소화(localization)*를 시각화.

train 분할로 특징 조합을 학습한 뒤, 테스트 발화 하나에 대해 겹쳐 그린다:
  - 스펙트로그램,
  - 정답 가짜 구간(빨강 음영),
  - 모델의 윈도우별 P(가짜)와 결정 임계값.

실행:  python3 analysis/localize_demo.py [feature] [uid]
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import os, sys
import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit

import ps_data as P
import pipeline as D
import features as F
import model as MD
from evaluate import eer


def main(feat="stft+phase+disc", uid=None, R=0.16, seed=0):
    grps = MD.FEATURE_SETS[feat]
    uids = D.sample_uids(R, 500, 750, 250, seed=seed)
    X, y, groups, _ = D.build(uids, R=R, cache=True)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    tr, te = next(gss.split(np.zeros(len(y)), y, groups))

    Xtr = MD.compose({g: X[g][tr] for g in grps}, grps)
    Xte = MD.compose({g: X[g][te] for g in grps}, grps)
    s, clf = MD.fit_predict(Xtr, y[tr], Xte)
    thr = eer(y[te], s)[1]

    # uid 미지정 시 테스트셋에서 부분가짜 발화 하나 선택
    test_groups = np.unique(groups[te])
    seg = P.load_seglab(R)
    if uid is None:
        for g in test_groups:
            u = uids[g]
            s = set(seg[u].tolist())
            if len(s) > 1:                       # 부분가짜
                uid = u
                break
    print(f"feature={feat}  uid={uid}  threshold={thr:.3f}")

    audio, sr = sf.read(os.path.join(P.WAV, uid + ".wav"))
    if audio.ndim > 1:
        audio = audio[:, 0]
    dur = len(audio) / sr
    labs = seg[uid]
    n = len(labs)
    feats_u = F.group_features(audio.astype(np.float32), n, R=R, groups=grps)
    prob = clf.predict_proba(MD.compose(feats_u, grps))[:, 1]
    gt_spoof = (np.asarray(labs) == "0").astype(int)
    t = (np.arange(n) + 0.5) * R

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    ax1.specgram(audio, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
    ax1.set_ylabel("freq (Hz)")
    ax1.set_title(f"{uid}  -  spoof localization ({feat}, R={R}s)   red = GT spoof")
    for s, e, lab in P.frames_to_intervals(labs.tolist(), R):
        if lab == "spoof":
            ax1.axvspan(s, min(e, dur), color="red", alpha=0.25)

    ax2.step(t, prob, where="mid", color="C0", label="P(spoof) predicted")
    ax2.fill_between(t, 0, gt_spoof, step="mid", color="red", alpha=0.2,
                     label="GT spoof (1=spoof)")
    ax2.axhline(thr, color="k", ls="--", lw=1, label=f"threshold={thr:.2f}")
    ax2.set_ylim(-0.02, 1.02); ax2.set_xlim(0, dur)
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("prob")
    ax2.legend(loc="upper right", fontsize=8)

    figdir = os.path.join(P.REPO, "figures")
    os.makedirs(figdir, exist_ok=True)
    out = os.path.join(figdir, f"localize_{feat.replace('+','-')}_{uid}.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print("saved:", out)
    return out


if __name__ == "__main__":
    feat = sys.argv[1] if len(sys.argv) > 1 else "stft"
    uid  = sys.argv[2] if len(sys.argv) > 2 else None
    main(feat, uid)
