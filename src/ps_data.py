#!/usr/bin/env python3
"""PartialSpoof (v1.2) dev-set parser / audio<->label verifier.

Dataset lives in the (git-ignored) upstream clone. Override with env PS_DATA.
  $PS_DATA/dev/con_wav/<id>.wav            audio
  $PS_DATA/dev/dev.lst                     list of utt ids
  $PS_DATA/protocols/PartialSpoof_LA_cm_protocols/PartialSpoof.LA.cm.dev.trl.txt
  $PS_DATA/segment_labels/dev_seglab_<res>.npy   {id: np.array([per-frame '0'/'1'])}

Label convention (verified): '1' = bonafide, '0' = spoof.
  LA_D_*  = original genuine ASVspoof2019 utterance  -> all '1'
  CON_D_* = concatenated utterance                   -> may be partially spoofed
"""
import os, sys
import numpy as np
import soundfile as sf

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
DB   = os.environ.get(
    "PS_DATA",
    os.path.join(REPO, "partialspoof", "database", "database"))
WAV  = os.path.join(DB, "dev", "con_wav")
SEG  = os.path.join(DB, "segment_labels")
PROTO = os.path.join(DB, "protocols",
                     "PartialSpoof_LA_cm_protocols",
                     "PartialSpoof.LA.cm.dev.trl.txt")

FRAME2STR = {"1": "bonafide", "0": "spoof"}
RESOLUTIONS = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]


def load_protocol():
    """utt_id -> (speaker, system, utt_label)"""
    out = {}
    with open(PROTO) as f:
        for line in f:
            spk, uid, _, sysid, lab = line.split()
            out[uid] = (spk, sysid, lab)
    return out


def load_seglab(res):
    path = os.path.join(SEG, f"dev_seglab_{res:.2f}.npy")
    return np.load(path, allow_pickle=True).item()


def frames_to_intervals(frames, res):
    """['1','0','0','1'] -> [(start_s, end_s, 'label'), ...] (run-length)."""
    out, i, n = [], 0, len(frames)
    while i < n:
        j = i
        while j < n and frames[j] == frames[i]:
            j += 1
        out.append((i * res, j * res, FRAME2STR[frames[i]]))
        i = j
    return out


def describe(uid, res=0.16):
    proto = load_protocol()
    seg = load_seglab(res)
    wav_path = os.path.join(WAV, uid + ".wav")
    audio, sr = sf.read(wav_path)
    dur = len(audio) / sr

    spk, sysid, ulab = proto[uid]
    frames = seg[uid].tolist()
    n = len(frames)

    print(f"=== {uid} ===")
    print(f"  wav         : {wav_path}")
    print(f"  sample_rate : {sr} Hz   samples: {len(audio)}   duration: {dur:.3f} s")
    print(f"  protocol    : speaker={spk}  system={sysid}  utt_label={ulab}")
    print(f"  seglab res  : {res} s   #frames: {n}   n*res={n*res:.3f} s "
          f"(ceil(dur/res)={int(np.ceil(dur/res))})")
    print(f"  frame labels: {''.join(frames)}")
    print(f"  intervals (start-end-label):")
    for s, e, lab in frames_to_intervals(frames, res):
        print(f"      {s:6.2f} - {e:6.2f} s   {lab}")
    return audio, sr, dur, frames


def crossres_check(uid):
    """frame count at each resolution should track duration."""
    wav_path = os.path.join(WAV, uid + ".wav")
    info = sf.info(wav_path)
    dur = info.frames / info.samplerate
    print(f"  duration = {dur:.3f}s")
    print(f"  {'res':>6} {'#frames':>8} {'n*res':>8} {'ceil(dur/res)':>14}")
    for r in RESOLUTIONS:
        seg = load_seglab(r)
        if uid not in seg:
            continue
        n = len(seg[uid])
        print(f"  {r:6.2f} {n:8d} {n*r:8.3f} {int(np.ceil(dur/r)):14d}")


if __name__ == "__main__":
    uid = sys.argv[1] if len(sys.argv) > 1 else "CON_D_0000000"
    res = float(sys.argv[2]) if len(sys.argv) > 2 else 0.16
    describe(uid, res)
    print("\n--- cross-resolution frame-count vs duration ---")
    crossres_check(uid)
