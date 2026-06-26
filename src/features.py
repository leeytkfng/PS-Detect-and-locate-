#!/usr/bin/env python3
"""[2][3][4] 프레이밍 + DSP 특징 추출 + 윈도우 풀링.

프레임 분석: 25 ms 윈도우 / 10 ms 홉 @ 16 kHz (n_fft=512, win=400).

특징 GROUP (각각 프레임별 시퀀스를 만든 뒤 라벨 윈도우로 풀링):
  magnitude(크기) -> STFT 로그 크기   (그룹 "stft")
                     LFCC + d + dd     (그룹 "lfcc")
  phase(위상)     -> 밴드별 순간주파수 편차 + 시간 위상 flux   (그룹 "phase")
  disc(불연속)    -> 밴드별 스펙트럼 flux + 로그에너지 d1/d2   (그룹 "disc")

이유: 가짜 구간은 '같은 화자'를 잘라 붙인 것이라 화자 단서는 무용지물이다.
크기는 보코더의 스펙트럼 질감을, 위상은 보코더의 위상 부정합을, 불연속은
이어붙인 이음새(seam)의 불연속을 잡는다. 이음새는 sparse(몇 프레임뿐)하므로
mean + std + MAX 로 풀링한다 -- max가 윈도우 내부의 불연속 peak를 보존한다.
"""
import numpy as np
import librosa
from scipy.fftpack import dct
from scipy.ndimage import median_filter
import parselmouth

SR        = 16000
WIN       = 400          # 25 ms 분석 윈도우
N_FFT     = 512          # fft 크기 (제로 패딩)
HOP       = 160          # 10 ms 홉
N_FILTERS = 40           # 선형 필터뱅크 채널 수 (LFCC)
N_CEPS    = 20           # 켑스트럼 계수 개수
N_BANDS   = 16           # phase/disc 밴드 묶음 개수
POOL_STATS = ("mean", "std", "max")

GROUPS = ("stft", "lfcc", "phase", "disc", "seam")

F0_FLOOR, F0_CEIL = 75, 500     # F0 탐색 범위 (Hz)


# ----------------------------------------------------------- 공용 STFT
def stft_complex(audio):
    return librosa.stft(audio, n_fft=N_FFT, win_length=WIN, hop_length=HOP,
                        window="hann", center=True)


# ----------------------------------------------------------- magnitude(크기) 그룹
def _linear_filterbank(n_filters=N_FILTERS, n_fft=N_FFT, sr=SR):
    n_bins = n_fft // 2 + 1
    edges = np.linspace(0, sr / 2, n_filters + 2)
    bf = np.linspace(0, sr / 2, n_bins)
    fb = np.zeros((n_filters, n_bins), np.float32)
    for m in range(1, n_filters + 1):
        lo, ctr, hi = edges[m - 1], edges[m], edges[m + 1]
        fb[m - 1] = np.clip(np.minimum((bf - lo) / (ctr - lo),
                                       (hi - bf) / (hi - ctr)), 0, None)
    return fb


_FB = _linear_filterbank()


def lfcc_seq(audio):
    P = np.abs(stft_complex(audio)) ** 2
    logfbe = np.log(_FB @ P + 1e-10)
    return dct(logfbe, type=2, axis=0, norm="ortho")[:N_CEPS].T     # (T, N_CEPS)


def add_deltas(seq):
    x = seq.T
    w = max(3, min(9, x.shape[1] if x.shape[1] % 2 else x.shape[1] - 1))
    d1 = librosa.feature.delta(x, order=1, width=w)
    d2 = librosa.feature.delta(x, order=2, width=w)
    return np.concatenate([x, d1, d2], axis=0).T                   # (T, 3*N_CEPS)


def logmag_seq(audio):
    return np.log(np.abs(stft_complex(audio)) + 1e-10).T.astype(np.float32)


# ----------------------------------------------------------- 밴드 헬퍼
def _band_matrix(n_bands=N_BANDS, n_bins=N_FFT // 2 + 1):
    edges = np.linspace(0, n_bins, n_bands + 1).astype(int)
    M = np.zeros((n_bands, n_bins), np.float32)
    for b in range(n_bands):
        lo, hi = edges[b], max(edges[b] + 1, edges[b + 1])
        M[b, lo:hi] = 1.0 / (hi - lo)
    return M


_BM = _band_matrix()


def _princarg(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


# ----------------------------------------------------------- phase(위상) 그룹
def phase_seq(audio):
    """(T, 2*N_BANDS): 밴드별 크기 가중 순간주파수 편차 + 위상 flux."""
    S = stft_complex(audio)
    mag = np.abs(S) + 1e-10
    phase = np.angle(S)
    n_bins = S.shape[0]
    dphi = np.diff(phase, axis=1)
    k = np.arange(n_bins)[:, None]
    expected = 2 * np.pi * HOP * k / N_FFT
    ifdev = np.abs(_princarg(dphi - expected))
    pflux = np.abs(_princarg(dphi))
    w = mag[:, 1:]
    ifd_b = _BM @ (ifdev * w) / (_BM @ w + 1e-10)
    pfl_b = _BM @ (pflux * w) / (_BM @ w + 1e-10)
    seq = np.concatenate([ifd_b, pfl_b], axis=0).T
    return np.vstack([seq[:1], seq]).astype(np.float32)


# ----------------------------------------------------------- disc(불연속) 그룹
def disc_seq(audio):
    """(T, N_BANDS+2): 밴드별 스펙트럼 flux + |로그에너지 d1| + |d2|."""
    S = np.abs(stft_complex(audio))
    Sn = S / (S.sum(axis=0, keepdims=True) + 1e-10)
    flux_b = (_BM @ np.abs(np.diff(Sn, axis=1))).T
    flux_b = np.vstack([flux_b[:1], flux_b])
    loge = np.log((S ** 2).sum(axis=0) + 1e-10)
    d1 = np.gradient(loge); d2 = np.gradient(d1)
    return np.concatenate([flux_b, np.abs(d1)[:, None],
                           np.abs(d2)[:, None]], axis=1).astype(np.float32)


# ----------------------------------------------------------- seam(이음새) 그룹
def _f0_aligned(audio, T):
    """STFT 프레임 격자(T)에 맞춘 F0(Hz)와 유성 플래그. parselmouth(Praat) 사용."""
    try:
        snd = parselmouth.Sound(np.ascontiguousarray(audio, np.float64), SR)
        pitch = snd.to_pitch(time_step=HOP / SR,
                             pitch_floor=F0_FLOOR, pitch_ceiling=F0_CEIL)
        f0 = pitch.selected_array["frequency"]      # 0 = 무성
        tp = np.asarray(pitch.xs())
    except Exception:
        return np.zeros(T, np.float32), np.zeros(T, np.float32)
    if len(f0) == 0:
        return np.zeros(T, np.float32), np.zeros(T, np.float32)
    tt = np.arange(T) * HOP / SR
    idx = np.clip(np.searchsorted(tp, tt), 0, len(f0) - 1)   # 최근접 정렬
    f0a = f0[idx].astype(np.float32)
    return f0a, (f0a > 0).astype(np.float32)


def seam_seq(audio):
    """(T, 4): 이음새 직격 특징 - 자연 음성의 '매끄러움'이 깨지는 지점을 잡는다.

      col0  |Δlog F0|       : 피치 궤적 점프 (유성 연속 구간에서만)
      col1  |Δvoiced|       : 유성<->무성 전환 (이음새가 VAD 경계와 겹침)
      col2  spectral novelty: 인접 프레임 음색(크기 스펙트럼) 코사인 거리
      col3  local novelty   : novelty - 로컬 중앙값 (날카로운 점프만 강조)
    """
    S = np.abs(stft_complex(audio)).T                # (T, bins)
    T = S.shape[0]

    f0, voiced = _f0_aligned(audio, T)
    logf0 = np.log(f0 + 1e-6)
    df0 = np.zeros(T, np.float32)
    df0[1:] = np.abs(np.diff(logf0)) * (voiced[1:] * voiced[:-1])
    vchg = np.zeros(T, np.float32)
    vchg[1:] = np.abs(np.diff(voiced))

    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-10)
    nov = np.zeros(T, np.float32)
    nov[1:] = 1.0 - np.sum(Sn[1:] * Sn[:-1], axis=1)     # 코사인 거리
    base = median_filter(nov, size=9, mode="nearest")
    novc = np.clip(nov - base, 0, None)                  # 국소 대비 (peak)

    return np.stack([df0, vchg, nov, novc], axis=1).astype(np.float32)


_SEQ_FN = {"stft": logmag_seq,
           "lfcc": lambda a: add_deltas(lfcc_seq(a)),
           "phase": phase_seq,
           "disc": disc_seq,
           "seam": seam_seq}


# ----------------------------------------------------------- [4] 풀링
def pool_to_windows(seq, n_windows, hop=HOP, sr=SR, R=0.16, stats=POOL_STATS):
    """(T, d) 프레임 시퀀스를 (n_windows, len(stats)*d)로 풀링.

    프레임 i(center=True)는 시각 i*hop/sr -> 윈도우 floor(time/R)에 속함.
    """
    T, d = seq.shape
    widx = np.floor(np.arange(T) * hop / sr / R).astype(int)
    out = np.zeros((n_windows, len(stats) * d), np.float32)
    for wi in range(n_windows):
        m = widx == wi
        if not m.any():
            if wi > 0:
                out[wi] = out[wi - 1]
            continue
        chunk = seq[m]
        parts = []
        for s in stats:
            parts.append(chunk.mean(0) if s == "mean" else
                         chunk.std(0) if s == "std" else chunk.max(0))
        out[wi] = np.concatenate(parts)
    return out


def group_features(audio, n_windows, R=0.16, groups=GROUPS, stats=POOL_STATS):
    """-> dict{그룹: (n_windows, *)} 풀링된 특징."""
    return {g: pool_to_windows(_SEQ_FN[g](audio), n_windows, R=R, stats=stats)
            for g in groups}


if __name__ == "__main__":
    import soundfile as sf, ps_data as P
    uid = "CON_D_0000000"
    a, _ = sf.read(f"{P.WAV}/{uid}.wav")
    n = len(P.load_seglab(0.16)[uid])
    for g, v in group_features(a.astype(np.float32), n).items():
        print(f"{g:6s} {v.shape}  nan={np.isnan(v).any()}")
