# Copyright (c) 2026 by Haizy Tiles
#Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Deeper per-track detail for the batch catalogue:

- Structure: intro, drops, breakdowns and outro, measured in bars.
- Sound: brightness, bass weight, stereo width, dynamics, tempo steadiness.
- Faults: clipping, dropouts and the metallic high-end shimmer Suno renders
  sometimes have, with the time each starts.

All of it is rough signal analysis. Treat it as a pointer to what to check.
"""

import numpy as np
import librosa

HOP = 512

COLUMNS = ["structure", "intro_bars", "outro_bars", "first_drop", "drops", "breakdowns", "dj_friendly",
           "brightness_hz", "bass_pct", "stereo_width_pct", "plr_db", "tempo_drift_pct", "faults"]

def _mmss(seconds):
    s = int(round(seconds))
    return f"{s // 60}:{s % 60:02d}"

def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-9))

# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------
def bar_levels(y, sr, bpm):
    """Loudness and kick-band loudness per bar, from the first strong beat."""
    if not bpm:
        return None, None, 0.0, 0.0
    bar = 4 * 60.0 / bpm
    onset = librosa.onset.onset_detect(y=y, sr=sr, hop_length=HOP, units="time")
    start = float(onset[0]) if len(onset) else 0.0
    n = int((len(y) / sr - start) // bar)
    if n < 8:
        return None, None, start, bar
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    t = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=HOP)
    full = (S ** 2).sum(axis=0)
    kick = (S[(freqs >= 35) & (freqs <= 120)] ** 2).sum(axis=0)
    edges = start + np.arange(n + 1) * bar
    idx = np.searchsorted(t, edges)
    lvl = np.array([full[idx[i]:idx[i + 1]].mean() if idx[i + 1] > idx[i] else 0 for i in range(n)])
    kik = np.array([kick[idx[i]:idx[i + 1]].mean() if idx[i + 1] > idx[i] else 0 for i in range(n)])
    return _db(np.sqrt(lvl)), _db(np.sqrt(kik)), start, bar

def structure(y, sr, bpm):
    """Columns for intro, drops, breakdowns and outro. A bar counts as 'full'
    when it is within 4 dB of the track's loud level and the kick is in."""
    out = {c: "" for c in COLUMNS[:7]}
    lvl, kik, start, bar = bar_levels(y, sr, bpm)
    if lvl is None:
        return out
    loud = np.percentile(lvl, 80)
    kick_on = kik > np.percentile(kik, 80) - 8
    full = (lvl > loud - 4) & kick_on
    # Smooth single odd bars so one fill does not split a section
    for i in range(1, len(full) - 1):
        if full[i - 1] == full[i + 1] != full[i]:
            full[i] = full[i - 1]
    # Runs of full / not full bars become sections
    runs, i = [], 0
    while i < len(full):
        j = i
        while j < len(full) and full[j] == full[i]:
            j += 1
        runs.append((bool(full[i]), i, j))
        i = j
    names = []
    drops = breaks = 0
    first_drop = None
    for k, (is_full, a, b) in enumerate(runs):
        length = b - a
        if is_full:
            names.append(f"drop {length}")
            drops += 1
            if first_drop is None:
                first_drop = start + a * bar
        elif k == 0:
            names.append(f"intro {length}")
        elif k == len(runs) - 1:
            names.append(f"outro {length}")
        else:
            names.append(f"break {length}")
            breaks += 1
    intro = runs[0][2] - runs[0][1] if not runs[0][0] else 0
    outro = runs[-1][2] - runs[-1][1] if not runs[-1][0] and len(runs) > 1 else 0
    out.update({
        "structure":   " · ".join(names),
        "intro_bars":  intro,
        "outro_bars":  outro,
        "first_drop":  _mmss(first_drop) if first_drop is not None else "",
        "drops":       drops,
        "breakdowns":  breaks,
        # DJs want at least 16 bars at each end to mix in and out
        "dj_friendly": "yes" if intro >= 16 and outro >= 16 else ("intro only" if intro >= 16 else
                       ("outro only" if outro >= 16 else "no")),
    })
    return out

# ---------------------------------------------------------------------------
# Sound
# ---------------------------------------------------------------------------
def sound(y, sr, native, native_sr, lufs, true_peak):
    out = {}
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=HOP)) ** 2
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    loud = S.sum(axis=0) > np.percentile(S.sum(axis=0), 30)
    if loud.any():
        cent = (freqs[:, None] * S[:, loud]).sum(axis=0) / np.maximum(S[:, loud].sum(axis=0), 1e-12)
        out["brightness_hz"] = int(np.median(cent))
        out["bass_pct"] = round(float(S[freqs < 150][:, loud].sum() / max(S[:, loud].sum(), 1e-12) * 100))
    if native.ndim == 2 and native.shape[0] == 2:
        mid, side = (native[0] + native[1]) / 2, (native[0] - native[1]) / 2
        out["stereo_width_pct"] = round(float(np.sqrt(np.mean(side ** 2)) / max(np.sqrt(np.mean(mid ** 2)), 1e-9) * 100))
    else:
        out["stereo_width_pct"] = 0
    if lufs != "" and true_peak != "":
        # Peak to loudness ratio: low means squashed, high means open and punchy
        out["plr_db"] = round(true_peak - lufs, 1)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP)
    times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
    if len(times) >= 16:
        # Tempo in 8-beat windows; drift is how far it wanders from the median
        win = [60 * 8 / (times[i + 8] - times[i]) for i in range(0, len(times) - 8, 4)]
        out["tempo_drift_pct"] = round(float((np.percentile(win, 90) - np.percentile(win, 10)) / np.median(win) * 100), 1)
    return out

# ---------------------------------------------------------------------------
# Faults
# ---------------------------------------------------------------------------
SHIMMER_DB = 15          # a 9-15 kHz tone this far above the band's average floor counts as ringing

def faults(native, native_sr):
    """Short text listing clipping, dropouts and shimmer, with start times."""
    found = []
    mono = native.mean(axis=0) if native.ndim == 2 else native
    chans = native if native.ndim == 2 else native[None, :]

    # Clipping: 3 or more samples in a row at full scale
    clip = (np.abs(chans) >= 0.999).any(axis=0)
    if clip.sum() >= 3:
        d = np.diff(np.r_[0, clip.astype(np.int8), 0])
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        runs = starts[(ends - starts) >= 3]
        if len(runs):
            found.append(f"clipping x{len(runs)} from {_mmss(runs[0] / native_sr)}")

    # Dropouts: sound vanishes for 50 ms to 1 s in the middle of loud music
    frame = int(native_sr * 0.05)
    n = len(mono) // frame
    if n > 40:
        rms = _db(np.sqrt(np.mean(mono[:n * frame].reshape(n, frame) ** 2, axis=1)))
        loud = np.percentile(rms, 70)
        quiet = rms < loud - 40
        gaps = []
        i = 20                                   # skip the first second
        while i < n - 20:                        # and the last
            if quiet[i]:
                j = i
                while j < n and quiet[j]:
                    j += 1
                if j - i <= 20 and rms[max(0, i - 4):i].mean() > loud - 10 and rms[j:j + 4].mean() > loud - 10:
                    gaps.append(i)
                i = j
            else:
                i += 1
        if gaps:
            found.append(f"dropout x{len(gaps)} at {_mmss(gaps[0] * frame / native_sr)}")

    # Shimmer: steady, ringing tones at 9-15 kHz that stand out of the usual
    # noise-like air. Averaging over the track smears real notes and cymbals,
    # so only peaks that ring for a long time stay well above the band's floor.
    if native_sr >= 32000:
        S = np.abs(librosa.stft(mono, n_fft=4096, hop_length=2048)) ** 2
        freqs = librosa.fft_frequencies(sr=native_sr, n_fft=4096)
        band = S[(freqs >= 9000) & (freqs <= 15000)]
        mids = S[(freqs > 200) & (freqs < 4000)]
        avg = band.mean(axis=1)
        if band.size and _db(np.sqrt(avg.mean())) > _db(np.sqrt(mids.mean())) - 50:
            peaks = np.flatnonzero(10 * np.log10(avg / np.median(avg)) > SHIMMER_DB)
            if len(peaks):
                strong = band[peaks].sum(axis=0) > np.median(band[peaks].sum(axis=0))
                first = int(np.argmax(strong))
                found.append(f"shimmer from {_mmss(first * 2048 / native_sr)}")
    return "; ".join(found)

def detail(y, sr, native, native_sr, bpm, lufs, true_peak):
    out = structure(y, sr, bpm)
    out.update(sound(y, sr, native, native_sr, lufs, true_peak))
    out["faults"] = faults(native, native_sr)
    return out
