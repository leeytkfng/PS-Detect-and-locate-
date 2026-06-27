#!/usr/bin/env python3
"""학습용 데이터 증강 (train만, 라벨/타이밍 보존). 외부 데이터 불필요.

부분조작은 0.16s 프레임 라벨이 있으므로 '시간축을 바꾸는 증강(속도/시간늘이기/
피치이동)'은 금지(라벨 어긋남). 아래 5종은 모두 타이밍 보존:

  noise    가산 백색잡음 (랜덤 SNR)
  reverb   합성 지수감쇠 임펄스응답 합성곱 (잔향/채널)
  codec    mu-law 8bit 압축-복원 (전화 코덱 프록시, LA 시나리오)
  rawboost RawBoost류: 선형 합성곱 + 임펄스성 + 정상 가산잡음 결합
  specaug  (특징단) 주파수 마스킹  -- audio가 아니라 feature에 적용
"""
import numpy as np

SR = 16000


def add_noise(a, rng, snr_db=None):
    snr = rng.uniform(8, 30) if snr_db is None else snr_db
    p_sig = np.mean(a ** 2) + 1e-12
    p_n = p_sig / (10 ** (snr / 10))
    return (a + rng.randn(len(a)).astype(np.float32) * np.sqrt(p_n)).astype(np.float32)


def add_reverb(a, rng):
    dur = int(rng.uniform(0.05, 0.25) * SR)
    decay = rng.uniform(0.02, 0.10) * SR
    rir = (rng.randn(dur) * np.exp(-np.arange(dur) / decay)).astype(np.float32)
    rir[0] = 1.0
    out = np.convolve(a, rir)[:len(a)]
    peak = np.max(np.abs(out)) + 1e-9
    return (out / peak * (np.max(np.abs(a)) + 1e-9)).astype(np.float32)


def codec_mulaw(a, rng, mu=255.0):
    x = a / (np.max(np.abs(a)) + 1e-9)
    comp = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    q = np.round((comp * 0.5 + 0.5) * mu) / mu * 2 - 1          # 8-bit 양자화
    dec = np.sign(q) * ((1 + mu) ** np.abs(q) - 1) / mu
    return (dec * (np.max(np.abs(a)) + 1e-9)).astype(np.float32)


def rawboost(a, rng):
    n = rng.randint(2, 8)                                       # 1) 선형 합성곱
    fir = rng.randn(n).astype(np.float32); fir[0] = 1.0
    a = np.convolve(a, fir / (np.sum(np.abs(fir)) + 1e-9))[:len(a)]
    a = a.copy()
    k = max(1, int(len(a) * rng.uniform(1e-4, 1e-3)))          # 2) 임펄스성 잡음
    idx = rng.randint(0, len(a), k)
    a[idx] += rng.uniform(-1, 1, k).astype(np.float32) * (np.max(np.abs(a)) + 1e-9)
    return add_noise(a, rng, snr_db=rng.uniform(10, 25))        # 3) 정상 가산잡음


AUDIO_AUG = {"noise": add_noise, "reverb": add_reverb,
             "codec": codec_mulaw, "rawboost": rawboost}


def augment_audio(a, rng):
    """5종(specaug 제외 4종 audio + 가끔 noise+codec 체인) 중 랜덤 적용."""
    a = a.astype(np.float32)
    keys = list(AUDIO_AUG)
    k = keys[rng.randint(len(keys))]
    return AUDIO_AUG[k](a, rng)


def specaug_freqmask(feat, rng, n_masks=2, max_w=0.15):
    """(특징단) 윈도우 특징 행렬에 주파수(=차원) 마스킹. 타이밍 보존."""
    out = feat.copy()
    d = out.shape[1]
    for _ in range(n_masks):
        w = int(rng.uniform(0, max_w) * d)
        if w < 1:
            continue
        s = rng.randint(0, max(1, d - w))
        out[:, s:s + w] = 0.0
    return out
