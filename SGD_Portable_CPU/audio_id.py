# Copyright (c) 2026 by Haizy Tiles
#Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Audio fingerprints with Chromaprint (fpcalc), the same fingerprints AcoustID
and MusicBrainz use.

- Duplicates: finds tracks in a batch that share the same audio, such as the
  same render saved twice, a WAV and its MP3, or a Suno extend that reuses
  the start of a take.
- Released music: optionally asks AcoustID whether a track matches a song
  that is already out. Needs a free AcoustID application key.
"""

import base64
import csv
import json
import os
import shutil
import subprocess
import time
import urllib.parse
import urllib.request

import numpy as np

PRINT_NAME   = "sgd_chromaprint.jsonl"
MATCHES_NAME = "sgd_known_matches.csv"
KEY_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "acoustid_key.txt")
LENGTH       = 120          # seconds fingerprinted from the start, like AcoustID
CODES_PER_S  = 8.06         # Chromaprint gives one 32-bit code about every 0.124 s
MIN_OVERLAP  = 10           # seconds of shared audio before two tracks count as the same
MIN_MATCH    = 0.80         # share of matching bits; unrelated audio sits near 0.5
COMMON_CODE  = 30           # codes found in more tracks than this are silence or noise

def _fpcalc():
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("fpcalc.exe", "fpcalc"):
        p = os.path.join(here, "bin", name)
        if os.path.exists(p):
            return p
    return shutil.which("fpcalc")

def available():
    return _fpcalc() is not None

def _run(args):
    res = subprocess.run(args, capture_output=True, timeout=120,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode("utf-8", errors="replace").strip()[:200] or "fpcalc failed")
    return json.loads(res.stdout.decode("utf-8", errors="replace"))

def fingerprint(path):
    """{duration, fp (compressed, for AcoustID), raw (base64 uint32 codes)}, or None."""
    exe = _fpcalc()
    if exe is None:
        return None
    try:
        packed = _run([exe, "-json", "-length", str(LENGTH), path])
        raw = _run([exe, "-json", "-raw", "-length", str(LENGTH), path])
    except Exception as e:
        print(f"Fingerprint failed for {os.path.basename(path)}: {str(e)[:200]}")
        return None
    codes = np.asarray(raw.get("fingerprint", []), dtype=np.int64).astype(np.uint32)
    return {"duration": round(float(packed.get("duration", 0)), 1),
            "fp": packed.get("fingerprint", ""),
            "raw": base64.b64encode(codes.tobytes()).decode("ascii")}

def save_print(root, row, fp):
    try:
        with open(os.path.join(root, PRINT_NAME), "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(file=row["file"], folder=row["folder"], **fp), ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"Could not save fingerprint for {row['file']}: {e}")

def read_prints(folder):
    """{track key: entry with 'codes' as a uint32 array}. The last line for a track wins."""
    out = {}
    path = os.path.join(folder, PRINT_NAME)
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            d["codes"] = np.frombuffer(base64.b64decode(d.get("raw", "")), dtype=np.uint32)
            key = os.path.normcase(os.path.normpath(os.path.join(d.get("folder") or ".", d.get("file", ""))))
            out[key] = d
    return out

# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------
_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)

def _match(a, b, offset):
    """Share of matching bits where a, shifted by offset, overlaps b."""
    if offset >= 0:
        a, b = a[offset:], b
    else:
        a, b = a, b[-offset:]
    n = min(len(a), len(b))
    if n == 0:
        return 0.0, 0
    x = np.bitwise_xor(a[:n], b[:n]).view(np.uint8)
    return 1 - _POP[x].sum() / (32 * n), n

def find_same_audio(prints):
    """[(key_a, key_b, match 0-1, seconds shared)] for tracks that share audio.
    Codes shared exactly between two tracks vote for a time offset; the best
    offset is then checked bit by bit."""
    keys = [k for k, d in prints.items() if len(d["codes"])]
    if len(keys) < 2:
        return []
    codes = np.concatenate([prints[k]["codes"] for k in keys])
    track = np.concatenate([np.full(len(prints[k]["codes"]), i, dtype=np.int32) for i, k in enumerate(keys)])
    pos   = np.concatenate([np.arange(len(prints[k]["codes"]), dtype=np.int32) for k in keys])
    order = np.argsort(codes, kind="stable")
    codes, track, pos = codes[order], track[order], pos[order]
    starts = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]])
    sizes  = np.diff(np.r_[starts, len(codes)])
    votes = {}
    for s, n in zip(starts[sizes > 1], sizes[sizes > 1]):
        t, p = track[s:s + n], pos[s:s + n]
        if len(set(t.tolist())) > COMMON_CODE:
            continue
        for i in range(n):
            for j in range(i + 1, n):
                a, b = int(t[i]), int(t[j])
                if a == b:
                    continue
                if a > b:
                    a, b, off = b, a, int(p[j] - p[i])
                else:
                    off = int(p[i] - p[j])
                votes[(a, b, off)] = votes.get((a, b, off), 0) + 1
    best = {}
    for (a, b, off), v in votes.items():
        if v >= 5 and v > best.get((a, b), (0, 0))[0]:
            best[(a, b)] = (v, off)
    found = []
    for (a, b), (_, off) in best.items():
        m, n = _match(prints[keys[a]]["codes"], prints[keys[b]]["codes"], off)
        secs = n / CODES_PER_S
        if m >= MIN_MATCH and secs >= MIN_OVERLAP:
            found.append((keys[a], keys[b], round(float(m), 2), round(secs)))
    return found

# ---------------------------------------------------------------------------
# AcoustID lookup
# ---------------------------------------------------------------------------
def read_key():
    try:
        with open(KEY_FILE, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""

def save_key(key):
    key = (key or "").strip()
    if key:
        with open(KEY_FILE, "w", encoding="utf-8") as f:
            f.write(key)
    return key

def lookup(key, duration, fp):
    """[(score, title, artist, acoustid)] for released recordings that match."""
    data = urllib.parse.urlencode({"client": key, "duration": int(duration), "fingerprint": fp,
                                   "meta": "recordings", "format": "json"}).encode()
    with urllib.request.urlopen("https://api.acoustid.org/v2/lookup", data=data, timeout=30) as r:
        res = json.loads(r.read().decode("utf-8"))
    if res.get("status") != "ok":
        raise RuntimeError(res.get("error", {}).get("message", "lookup failed"))
    out = []
    for hit in res.get("results", []):
        for rec in hit.get("recordings", []) or [{}]:
            artist = ", ".join(a.get("name", "") for a in rec.get("artists", []))
            out.append((round(hit.get("score", 0), 2), rec.get("title", ""), artist, hit.get("id", "")))
    return out

def check_released(folder, key, stop=None):
    """Look every fingerprinted track up on AcoustID, about 3 a second.
    Yields progress lines and writes sgd_known_matches.csv."""
    prints = read_prints(folder)
    if not prints:
        yield f"No {PRINT_NAME} in this folder. Run a batch first."
        return
    rows, errors = [], 0
    yield f"Checking {len(prints)} track(s) against AcoustID..."
    for i, d in enumerate(prints.values(), 1):
        if stop is not None and stop.is_set():
            yield f"Stopped after {i - 1} track(s)."
            break
        try:
            hits = lookup(key, d.get("duration", 0), d.get("fp", ""))
        except Exception as e:
            errors += 1
            if errors == 1 or errors % 50 == 0:
                yield f"Lookup failed for {d['file']}: {str(e)[:150]}"
            if "invalid api key" in str(e).lower():
                yield "AcoustID says the key is wrong. Check it and try again."
                return
            hits = []
        for score, title, artist, acoustid in hits:
            if score >= 0.5:
                rows.append({"file": d["file"], "folder": d.get("folder", ""), "match": score,
                             "title": title, "artist": artist, "acoustid": acoustid})
        if i % 100 == 0:
            yield f"[{i}/{len(prints)}] {len({(r['folder'], r['file']) for r in rows})} track(s) matched so far."
        time.sleep(0.34)
    path = os.path.join(folder, MATCHES_NAME)
    try:
        with open(path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["file", "folder", "match", "title", "artist", "acoustid"])
            w.writeheader()
            w.writerows(rows)
    except OSError as e:
        yield f"Could not write {MATCHES_NAME}: {e}. Close it in Excel and try again."
        return
    n = len({(r["folder"], r["file"]) for r in rows})
    yield (f"Done. {n} track(s) match something already released. See {MATCHES_NAME}."
           if n else "Done. No track matches anything AcoustID knows.")
