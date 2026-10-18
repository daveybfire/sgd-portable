# Copyright (c) 2026 by Haizy Tiles
#Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Track facts for the batch catalogue: BPM, key (with Camelot code), length,
loudness, energy, how the track ends, the idea it belongs to and its lane.
Everything here works on the mono 16 kHz signal SGD already loads.
"""

import csv
import os
import re

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------
HOP = 512

def estimate_bpm(y, sr):
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=HOP)
    _, beats = librosa.beat.beat_track(onset_envelope=env, sr=sr, hop_length=HOP, start_bpm=120.0)
    times = librosa.frames_to_time(beats, sr=sr, hop_length=HOP)
    if len(times) < 8:
        return None
    # A straight line through the beat times is far more precise than the
    # tracker's own tempo grid. Fall back to the median gap if the beats drift.
    ibi_bpm = 60 / np.median(np.diff(times))
    fit_bpm = 60 / np.polyfit(np.arange(len(times)), times, 1)[0]
    bpm = fit_bpm if abs(fit_bpm - ibi_bpm) / ibi_bpm < 0.04 else ibi_bpm
    # Beat trackers often land on half or double time. Fold into 80-180.
    while bpm < 80:
        bpm *= 2
    while bpm > 180:
        bpm /= 2
    return round(float(bpm), 1)

# The beat tracker sometimes locks onto a half, double, 2/3 or 3/4 pulse.
# When the track's top style says where the tempo should sit, move it there.
STYLE_BPM = [
    ((115, 135), {"house", "deep house", "tech house", "techno", "minimal", "minimal techno",
                  "deep techno", "progressive house", "acid house", "tribal house", "electro house",
                  "garage house", "ghetto house", "euro house", "hip-house", "tropical house",
                  "nu-disco", "disco", "uk garage", "speed garage", "bassline", "uk funky",
                  "italo house", "hard house", "dance-pop", "eurodance"}),
    ((160, 180), {"drum n bass", "jungle", "juke", "footwork"}),
    ((135, 150), {"dubstep", "halftime", "grime", "trap"}),
]
BPM_STEPS = (2, 0.5, 1.5, 4 / 3, 0.75, 2 / 3)

def fit_bpm_to_style(bpm, style_scores):
    if bpm is None or not style_scores:
        return bpm
    top = max(style_scores.items(), key=lambda x: x[1])[0].lower()
    for (lo, hi), styles in STYLE_BPM:
        if top in styles:
            if lo <= bpm <= hi:
                return bpm
            for step in BPM_STEPS:
                if lo <= bpm * step <= hi:
                    return round(bpm * step, 1)
    return bpm

# ---------------------------------------------------------------------------
# Key (Krumhansl-Schmuckler profiles on the average chroma)
# ---------------------------------------------------------------------------
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Camelot wheel codes used for harmonic mixing, indexed by tonic
CAMELOT_MAJOR = ["8B", "3B", "10B", "5B", "12B", "7B", "2B", "9B", "4B", "11B", "6B", "1B"]
CAMELOT_MINOR = ["5A", "12A", "7A", "2A", "9A", "4A", "11A", "6A", "1A", "8A", "3A", "10A"]

def estimate_key(y, sr):
    """Return (key name, Camelot code), e.g. ("A minor", "8A")."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=HOP).mean(axis=1)
    if not np.any(chroma):
        return "", ""
    best = (-2.0, 0, "major")
    for tonic in range(12):
        rolled = np.roll(chroma, -tonic)
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(rolled, profile)[0, 1]
            if r > best[0]:
                best = (r, tonic, mode)
    _, tonic, mode = best
    camelot = (CAMELOT_MAJOR if mode == "major" else CAMELOT_MINOR)[tonic]
    return f"{NOTES[tonic]} {mode}", camelot

# ---------------------------------------------------------------------------
# Loudness, energy and ending
# ---------------------------------------------------------------------------
def _db(x):
    return 20 * np.log10(np.maximum(x, 1e-9))

def loudness_and_energy(y, sr=16000):
    """Average loudness in dBFS (RMS) and a rough 1-10 energy score.
    Energy uses the loud parts (top 10% of half-second frames), so a quiet
    intro or break does not drag a banging track down."""
    rms_db = float(_db(np.sqrt(np.mean(y ** 2))))
    frame = sr // 2
    n = len(y) // frame
    if n >= 4:
        frames = np.sqrt(np.mean(y[:n * frame].reshape(n, frame) ** 2, axis=1))
        loud_db = float(_db(np.percentile(frames, 90)))
    else:
        loud_db = rms_db
    # -26 dBFS (quiet) maps to 1, -6 dBFS (loud master) maps to 10
    energy = int(round(np.clip((loud_db + 26) / 20 * 9 + 1, 1, 10)))
    return round(rms_db, 1), energy

def lufs_and_peak(audio, sr):
    """Integrated loudness in LUFS, the measure streaming services use, and
    true peak in dBTP. audio is (samples,) or (channels, samples) at the file's
    own sample rate. Blank if pyloudnorm is missing."""
    try:
        import pyloudnorm
        from scipy.signal import resample_poly
    except ImportError:
        return "", ""
    data = audio.T if audio.ndim == 2 else audio
    try:
        lufs = pyloudnorm.Meter(sr).integrated_loudness(data)
    except ValueError:                          # shorter than one 0.4 s block
        return "", ""
    # True peak: oversample 4x, a minute at a time to keep memory low
    chans = audio if audio.ndim == 2 else audio[None, :]
    step, peak = sr * 60, 0.0
    for start in range(0, chans.shape[1], step):
        piece = chans[:, max(0, start - 64):start + step + 64]
        peak = max(peak, float(np.max(np.abs(resample_poly(piece, 4, 1, axis=1)))))
    lufs = round(float(lufs), 1) if np.isfinite(lufs) else ""
    return lufs, round(float(_db(peak)), 1)

def ending_type(y, sr):
    """clean, fade, cut off or long silence. A rough check, not a verdict."""
    frame = int(sr * 0.1)
    n = len(y) // frame
    if n < 40:
        return ""
    rms = np.sqrt(np.mean(y[:n * frame].reshape(n, frame) ** 2, axis=1))
    loud = np.median(rms[rms > 0]) if np.any(rms > 0) else 0
    if loud == 0:
        return "silent"
    quiet = rms < loud * 0.01
    tail_silence = 0
    for q in quiet[::-1]:
        if not q:
            break
        tail_silence += 1
    if tail_silence >= 40:                     # 4 s or more of silence at the end
        return "long silence"
    if np.mean(rms[-3:]) > loud * 0.3:         # still loud in the last 0.3 s
        return "cut off"
    sound = rms[: n - tail_silence]
    if len(sound) >= 80:
        before = np.mean(sound[-80:-40])       # 8-4 s before the sound stops
        after  = np.mean(sound[-40:])          # last 4 s of sound
        if after < before * 0.5 and before > loud * 0.3:
            return "fade"
    return "clean"

# ---------------------------------------------------------------------------
# Idea name: groups takes such as "ACID CELLULOID [E] 001" and "... 004"
# ---------------------------------------------------------------------------
def idea_name(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    name = re.sub(r"\[[^\]]*\]|\([^)]*\)", " ", name)        # [E], [s], (remix)
    name = re.sub(r"_\d+\b", " ", name)                      # _001
    name = re.sub(r"\bv\d+\b", " ", name, flags=re.I)        # v3
    name = re.sub(r"[\s_-]+\d+\s*$", " ", name)              # trailing take number
    name = re.sub(r"\s+", " ", name).strip(" -_")
    return name or os.path.splitext(os.path.basename(filename))[0]

# ---------------------------------------------------------------------------
# Lanes: one per artist name, defined in lanes.csv next to this file
# ---------------------------------------------------------------------------
LANES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lanes.csv")
UNSORTED = "Unsorted"
LANE_MIN_SCORE = 0.10

_lanes = None
_lanes_mtime = None

def _num(text):
    try:
        return float(text)
    except (TypeError, ValueError):
        return None

def load_lanes():
    """Read lanes.csv again whenever it changes, so edits apply on the next track."""
    global _lanes, _lanes_mtime
    try:
        mtime = os.path.getmtime(LANES_FILE)
    except OSError:
        return []
    if _lanes is not None and mtime == _lanes_mtime:
        return _lanes
    with open(LANES_FILE, "r", encoding="utf-8-sig", newline="") as f:
        text = f.read()
    delim = ";" if text.split("\n", 1)[0].count(";") > text.split("\n", 1)[0].count(",") else ","
    lanes = []
    for row in csv.DictReader(text.splitlines(), delimiter=delim):
        lane = (row.get("lane") or "").strip()
        if not lane:
            continue
        lanes.append({
            "lane":    lane,
            "artist":  (row.get("artist_name") or "").strip(),
            "styles":  {s.strip().lower() for s in re.split(r"[;|]", row.get("styles") or "") if s.strip()},
            "bpm_min": _num(row.get("bpm_min")),
            "bpm_max": _num(row.get("bpm_max")),
            "weight":  _num(row.get("weight")) or 1.0,
        })
    _lanes, _lanes_mtime = lanes, mtime
    return lanes

def pick_lane(style_scores, bpm):
    """(lane, artist name) whose styles score highest for this track, within its BPM range."""
    best_lane, best_artist, best_score = UNSORTED, "", LANE_MIN_SCORE
    for lane in load_lanes():
        if bpm is not None:
            if lane["bpm_min"] is not None and bpm < lane["bpm_min"]:
                continue
            if lane["bpm_max"] is not None and bpm >= lane["bpm_max"]:
                continue
        score = lane["weight"] * max((s for st, s in style_scores.items() if st.lower() in lane["styles"]), default=0)
        if score > best_score:
            best_lane, best_artist, best_score = lane["lane"], lane["artist"], score
    return best_lane, best_artist

# ---------------------------------------------------------------------------
# All facts for one track
# ---------------------------------------------------------------------------
def track_facts(y, sr, filename, style_scores=None):
    bpm = fit_bpm_to_style(estimate_bpm(y, sr), style_scores)
    key, camelot = estimate_key(y, sr)
    _, energy = loudness_and_energy(y, sr)
    return {
        "idea":        idea_name(filename),
        "bpm":         bpm if bpm is not None else "",
        "key":         key,
        "camelot":     camelot,
        "length":      f"{int(len(y) / sr) // 60}:{int(len(y) / sr) % 60:02d}",
        "energy":      energy,
        "ending":      ending_type(y, sr),
    }
