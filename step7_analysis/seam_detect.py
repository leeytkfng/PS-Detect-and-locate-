#!/usr/bin/env python3
"""이음새(seam) 경계 검출기 - "robust" 버전.

raw 버전이 약했던 이유를 정조준해서 고친다:
  - F0 점프가 랜덤 이하였던 건 대부분 피치트래커 octave 오류 -> octave 보정 + median 평활
  - 단일 글로벌 피크만 봐서 약했음 -> 임계값 넘는 모든 피크 + 다중 단서 융합

경계 점수(boundary score) = 세 단서의 '국소 대비(local-contrast)' z-점수 합:
  1) |Δ semitone F0|  (octave 보정·median 평활, 유성 연속 구간)
  2) 스펙트럼 novelty  (인접 프레임 크기 스펙트럼 코사인 거리)
  3) 위상 불연속       (phase IF-deviation 곡선의 시간 변화)

평가: 검출 피크 vs 실제 이음새(세그먼트 라벨 전환)를 ±tol 안에서 매칭하여
precision / recall / F1. 단일 피크 적중률이 아니라 제대로 된 검출 평가.

실행:  python3 src/seam_detect.py            # 300개 부분가짜로 P/R/F1
       python3 src/seam_detect.py plot <uid> # 한 발화 경계점수 그림
"""
import os as _os, sys as _sys
_sys.path[:0] = [f.path for f in _os.scandir(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    if f.is_dir() and (f.name.startswith("step") or f.name == "tools")]
import os, sys
import numpy as np
import soundfile as sf
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

import ps_data as P
import features as F
import pipeline as PL


# ----------------------------------------------------------- robust F0
def robust_semitone(audio, T):
    """STFT 격자(T)에 정렬된 semitone F0. octave 보정 + median 평활. 무성=nan."""
    f0, voiced = F._f0_aligned(audio, T)
    s = np.full(T, np.nan, np.float32)
    m = voiced > 0
    if m.sum() < 3:
        return s
    s[m] = 12.0 * np.log2(f0[m] / 55.0)                 # 55 Hz 기준 semitone
    # 결측(무성) 보간 후 median 평활 -> 미세 지터 제거, 그 뒤 무성 마스크 복원
    filled = np.interp(np.arange(T), np.where(m)[0], s[m])
    sm = median_filter(filled, size=5, mode="nearest")
    s = np.where(m, sm, np.nan).astype(np.float32)
    # octave 보정: 인접 유성 프레임 점프가 6 semitone 초과면 옥타브 단위로 스냅
    for i in range(1, T):
        if np.isnan(s[i]) or np.isnan(s[i - 1]):
            continue
        while s[i] - s[i - 1] > 6:
            s[i] -= 12
        while s[i] - s[i - 1] < -6:
            s[i] += 12
    return s


def _contrast(x, w=9):
    """국소 대비: x - 로컬 중앙값, 음수 절단, 단위 표준편차 정규화."""
    c = np.clip(x - median_filter(x, size=w, mode="nearest"), 0, None)
    sd = c.std()
    return c / (sd + 1e-9)


# ----------------------------------------------------------- 경계 점수
def boundary_components(audio):
    """-> (times, dict{f0,novelty,phase} 국소대비 곡선)."""
    S = np.abs(F.stft_complex(audio)).T                 # (T, bins)
    T = S.shape[0]
    times = np.arange(T) * F.HOP / F.SR

    # 1) F0 점프 (semitone, 유성 연속 구간만)
    s = robust_semitone(audio, T)
    df0 = np.zeros(T, np.float32)
    for i in range(1, T):
        if not (np.isnan(s[i]) or np.isnan(s[i - 1])):
            df0[i] = abs(s[i] - s[i - 1])

    # 2) 스펙트럼 novelty
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-10)
    nov = np.zeros(T, np.float32)
    nov[1:] = 1.0 - np.sum(Sn[1:] * Sn[:-1], axis=1)

    # 3) 위상 불연속 (IF-deviation 곡선의 시간 변화)
    ph = F.phase_seq(audio)[:, :F.N_BANDS].mean(axis=1)  # 평균 IF-deviation
    pd = np.zeros(T, np.float32)
    pd[1:] = np.abs(np.diff(ph))

    return times, {"f0": _contrast(df0), "novelty": _contrast(nov),
                   "phase": _contrast(pd)}


def boundary_score(audio, weights=(1.0, 1.0, 1.0)):
    times, comp = boundary_components(audio)
    wf, wn, wp = weights
    score = wf * comp["f0"] + wn * comp["novelty"] + wp * comp["phase"]
    return times, score, comp


def detect(times, score, thr=2.0, min_gap_s=0.10):
    """임계값 넘는 피크들을 경계 후보로. -> 피크 시각 배열."""
    dist = max(1, int(min_gap_s * F.SR / F.HOP))
    pk, _ = find_peaks(score, height=thr, distance=dist)
    return times[pk]


# ----------------------------------------------------------- 평가
def gt_seams(uid, R=0.02):
    iv = P.frames_to_intervals(P.load_seglab(R)[uid].tolist(), R)
    return np.array([s for s, _, _ in iv[1:]])          # 내부 경계만


def _match(pred, gt, tol):
    """그리디 최근접 매칭 -> (tp)."""
    if len(pred) == 0 or len(gt) == 0:
        return 0
    used = np.zeros(len(gt), bool)
    tp = 0
    for p in pred:
        d = np.abs(gt - p)
        d[used] = np.inf
        j = d.argmin()
        if d[j] <= tol:
            used[j] = True
            tp += 1
    return tp


def evaluate(n=300, R=0.02, tol=0.05, thr=2.0, seed=1, weights=(1, 1, 1),
             only=None):
    """부분가짜 n개에서 경계 검출 P/R/F1. only='novelty'면 단일 단서만."""
    uids = PL.sample_uids(R, 0, n, 0, seed=seed)
    TP = FP = NG = 0
    for u in uids:
        a, _ = sf.read(os.path.join(P.WAV, u + ".wav"))
        a = a.astype(np.float32)
        times, comp = boundary_components(a)
        if only:
            score = comp[only]
        else:
            score = sum(w * comp[k] for w, k in zip(weights, ("f0", "novelty", "phase")))
        pred = detect(times, score, thr=thr)
        gt = gt_seams(u, R)
        tp = _match(pred, gt, tol)
        TP += tp; FP += len(pred) - tp; NG += len(gt)
    prec = TP / (TP + FP + 1e-9)
    rec = TP / (NG + 1e-9)
    f1 = 2 * prec * rec / (prec + rec + 1e-9)
    return dict(prec=prec, rec=rec, f1=f1, TP=TP, FP=FP, GT=NG)


def main_eval():
    print(f"# 이음새 경계 검출 P/R/F1 (부분가짜 300개, ±50ms, thr=2.0)\n")
    print(f"  {'설정':<22}{'precision':>10}{'recall':>9}{'F1':>8}")
    rows = [("novelty 단독", dict(only="novelty")),
            ("phase 단독", dict(only="phase")),
            ("F0(robust) 단독", dict(only="f0")),
            ("융합 (F0+nov+phase)", dict()),
            ("융합, F0 가중2x", dict(weights=(2, 1, 1)))]
    for name, kw in rows:
        r = evaluate(**kw)
        print(f"  {name:<22}{r['prec']:>10.3f}{r['rec']:>9.3f}{r['f1']:>8.3f}")
    print("\n  (검출 피크가 실제 이음새 ±50ms 안이면 정답. raw 단일피크 9.7%와 비교.)")


def main_plot(uid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    a, sr = sf.read(os.path.join(P.WAV, uid + ".wav"))
    a = a.astype(np.float32)
    times, score, comp = boundary_score(a)
    pred = detect(times, score)
    gt = gt_seams(uid)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axes[0].specgram(a, NFFT=512, Fs=sr, noverlap=384, cmap="magma")
    axes[0].set_ylabel("Hz")
    axes[0].set_title(f"{uid}  이음새 경계 검출  (점선=실제 이음새, ▲=검출)")
    axes[1].plot(times, comp["f0"], label="F0 점프", lw=.8)
    axes[1].plot(times, comp["novelty"], label="novelty", lw=.8)
    axes[1].plot(times, comp["phase"], label="위상", lw=.8)
    axes[1].plot(times, score, label="융합 점수", color="k", lw=1.2)
    for g in gt:
        for ax in axes:
            ax.axvline(g, color="r", ls="--", lw=1)
    axes[1].plot(pred, [score.max()] * len(pred), "k^", ms=8)
    axes[1].axhline(2.0, color="gray", ls=":", lw=.8)
    axes[1].legend(fontsize=8, ncol=4); axes[1].set_xlabel("time (s)")
    axes[1].set_xlim(0, len(a) / sr)
    out = os.path.join(P.REPO, "figures", f"seamdetect_{uid}.png")
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print("saved:", out)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        main_plot(sys.argv[2] if len(sys.argv) > 2 else "CON_D_0000000")
    else:
        main_eval()
