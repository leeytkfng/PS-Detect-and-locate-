# PS-Detect-and-locate

Partial-spoof (PS) **detection & localization** on the
[PartialSpoof](https://github.com/nii-yamagishilab/partialspoof) database (v1.2).

A partial spoof embeds short synthetic/edited speech segments into otherwise
genuine human audio. The goal here is to use **DSP features + a simple model**
to (a) decide whether an utterance contains spoofed speech and (b) **localize
which time spans are spoofed**.

## Dataset (dev subset used here)

Downloaded from Zenodo record `5766198` via the upstream `01_download_database.sh`.
We use the **dev** split only (train/eval are ~5–6 GB each and skipped for now).

| source | content |
|---|---|
| `dev/con_wav/<id>.wav` | 24,844 utterances, 16 kHz mono |
| `protocols/.../PartialSpoof.LA.cm.dev.trl.txt` | utterance label (`bonafide`/`spoof`) |
| `segment_labels/dev_seglab_<res>.npy` | per-frame `0/1` labels at 7 resolutions (0.01–0.64 s) |

**Verified facts**
- Frame label convention: **`1` = bonafide, `0` = spoof**.
- IDs: `LA_D_*` = original genuine ASVspoof2019 utterance (all `1`);
  `CON_D_*` = concatenated (may be partially spoofed).
- dev counts agree across all four sources: **24,844 total = 2,548 bonafide + 22,296 spoof**
  (of the 22,296 spoof: 9,900 fully spoofed, 12,396 *partial* spoof).
- Segment-label `.npy` is a 0-d object array wrapping `defaultdict{utt_id: array}`;
  `#frames == ceil(duration / resolution)` at every resolution.

> Note: the dataset itself is **not** committed (see `.gitignore`); it lives in the
> local `partialspoof/` clone. Set `PS_DATA` to point the code at another copy.

## Layout

```
src/        pipeline (ps_data, features, pipeline, model, evaluate, run) + make_report
analysis/   exploratory / figure scripts (label verification, seam DSP, ...)
figures/    generated plots
results/    saved experiment outputs
report/     PDF report
```

## Pipeline

```
[1] audio (16kHz wav)
[2] framing (25ms / 10ms hop)
[3] DSP features (per frame)
    A. magnitude : STFT log-mag, LFCC(+d+dd)
    B. phase     : IF-deviation, phase flux
    C. discontinuity : spectral flux, log-energy d2
[4] window pooling (0.16s, mean+std+max)  -> 1:1 with segment labels
[5] light classifier (LogReg -> LightGBM)
[6] window P(spoof):  localization = sequence,  detection = max pool
[7] post-proc (median smoothing) + eval (EER / Range-EER)
```

| module | role | step |
|---|---|---|
| `src/ps_data.py` | data access, protocol/label loaders | [1] |
| `src/features.py` | framing + DSP feature groups + pooling | [2][3][4] |
| `src/pipeline.py` | sampling + window dataset build | [1][2][4] |
| `src/model.py` | classifier, detection max-pool, smoothing | [5][6][7] |
| `src/evaluate.py` | EER metrics (+ Range-EER stub) | [7] |
| `src/run.py` | end-to-end orchestration | [1]->[7] |

`analysis/` holds exploratory/figure scripts (label verification, resolution
comparison, seam DSP analysis). `src/make_report.py` builds the PDF report.

## Quick start

```bash
python3 src/run.py                 # full pipeline, compares feature sets
python3 src/run.py --smooth 5      # with median smoothing
python3 src/ps_data.py CON_D_0000000 0.16   # inspect one utterance
```

## Results (dev, simple logistic-regression baseline)

| feature set | win-EER% | win-EER% +smooth | utt-EER% |
|---|---:|---:|---:|
| lfcc | 21.1 | 18.1 | 22.0 |
| stft | 18.1 | 15.2 | 9.6 |
| stft+phase+disc | 16.1 | 13.6 | 9.6 |
| **full** (stft+lfcc+phase+disc) | **14.3** | **12.4** | 11.1 |

Phase + discontinuity add to STFT (localization), and median smoothing helps
further. Detection (utterance) is best with STFT. Still a simple-regression
baseline; next: official **Range-EER**, finer resolution, LightGBM/CNN.
See **[RESULTS.md](RESULTS.md)**.
