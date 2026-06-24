# Results — Partial-Spoof Detection & Localization

First-pass DSP baseline comparing three feature families on the PartialSpoof
**dev** split. Reproduce with:

```bash
python3 src/experiment.py            # full run (1500 utts, R=0.16s, seed=0)
python3 src/localize_demo.py stft    # localization figure
```

## 1. What features, and why

| feature | dim (mean+std) | what it captures | why for PS |
|---|---|---|---|
| **LFCC** | 40 | linear-filterbank cepstrum (20 ceps) | linear (not mel) scale keeps **high-freq** detail where vocoder/synthesis artifacts live; standard ASVspoof front-end |
| **LFCC + Δ + ΔΔ** | 120 | LFCC + 1st/2nd time derivatives | adds **temporal dynamics** — unnatural smoothness of synthetic speech and abrupt **concatenation seams** |
| **STFT spectrogram** | 514 | raw log-magnitude STFT (257 bins) | full spectro-temporal pattern with **no cepstral compression** (higher-dim upper baseline) |

Analysis frames: `n_fft=512` (32 ms), `hop=160` (10 ms). Short-frame features are
**mean+std pooled** into 0.16 s windows aligned 1:1 with the ground-truth segment
labels, so every window has both a feature vector and a 0/1 label.

## 2. How detection/localization works

- **Localization** = window-level binary classification (spoof vs bonafide).
  One `LogisticRegression` (`class_weight="balanced"`, standardized features) per
  feature family predicts `P(spoof)` for each 0.16 s window → a spoof probability
  curve over time.
- **Detection** = utterance score = **max** window `P(spoof)`; an utterance is
  flagged spoof if any window looks spoofed.
- **Split**: `GroupShuffleSplit` by **utterance** (70/30) → no utterance leaks
  between train and test. Threshold taken at the EER operating point.

## 3. Evaluation results

Dev sample: **1500 utts** (500 bonafide, 750 partial-spoof, 250 full-spoof) →
**32,800 windows**, spoof-window rate 0.441. Test = 30 % of utts, split by id.

| feature | dim | win-EER % | win-AUC | win-F1 | win-bAcc | utt-EER % | utt-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| LFCC | 40 | 22.36 | 0.842 | 0.756 | 0.776 | 21.31 | 0.878 |
| LFCC+Δ+ΔΔ | 120 | 21.21 | 0.861 | 0.768 | 0.788 | 20.64 | 0.881 |
| **STFT-spec** | 514 | **16.94** | **0.905** | **0.814** | **0.831** | **9.40** | **0.971** |

*win-\* = localization (per 0.16 s window); utt-\* = detection (per utterance).*

**Takeaways**
- All three features work well above chance (EER ≪ 50 %), confirming the labels
  and pipeline are sound.
- Δ+ΔΔ consistently improve over plain LFCC (temporal cues matter).
- The raw STFT spectrogram is the strongest here — utterance detection EER
  **9.4 %**, AUC **0.97** — at the cost of much higher dimensionality.

Localization example (held-out `CON_D_0000022`): predicted `P(spoof)` rises into
the decision region exactly over the ground-truth spoof span.
See [figures/localize_stft_CON_D_0000022.png](figures/localize_stft_CON_D_0000022.png).

## 4. Limitations / next steps

- Trains/tests on a **subsample of dev** split by utterance (the official `train`
  split is not downloaded yet). Numbers are a baseline, not the paper protocol.
- Linear classifier + mean/std pooling is deliberately simple. Likely gains:
  finer resolution (0.02 s), GMM/LightGBM/CNN back-ends, score smoothing for
  cleaner spans, and proper train→dev→eval evaluation.
