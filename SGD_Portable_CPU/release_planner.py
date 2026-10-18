# Copyright (c) 2026 by Haizy Tiles
#Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Release planning from a finished batch catalogue.

- sgd_listen_first.csv: the best take of every idea, best first, then the
  other takes. Work down it from the top.
- sgd_releases.csv: suggested EPs or albums, one lane at a time. Tracks in
  a release sound alike, sit in a similar BPM range, mix well by key, never
  share an idea, and are put in an order with a rise and fall in energy.

Both files are plain suggestions to edit in Excel.
"""

import csv
import json
import math
import os
from datetime import datetime

import numpy as np

LISTEN_NAME   = "sgd_listen_first.csv"
RELEASES_NAME = "sgd_releases.csv"
BPM_TOLERANCE = 0.05        # tracks within 5% of the release's BPM
MIN_RELEASE   = 3           # fewer tracks than this stay unplaced

# ---------------------------------------------------------------------------
# Reading the catalogue
# ---------------------------------------------------------------------------
def _num(text, default=None):
    try:
        return float(text)
    except (TypeError, ValueError):
        return default

def _key(folder, file):
    return os.path.normcase(os.path.normpath(os.path.join(folder or ".", file)))

def read_tracks(folder, catalogue_name, fingerprint_name):
    """Tracks that analysed fine, newest row per file, with fingerprints attached."""
    rows = {}
    with open(os.path.join(folder, catalogue_name), "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("file"):
                rows[_key(row.get("folder"), row["file"])] = row
    prints = {}
    fp_path = os.path.join(folder, fingerprint_name)
    if os.path.exists(fp_path):
        with open(fp_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                prints[_key(d.get("folder"), d.get("file", ""))] = d.get("scores", {})
    tracks = []
    for k, row in rows.items():
        if row.get("status") != "ok":
            continue
        t = dict(row)
        t["_bpm"]     = _num(row.get("bpm"))
        t["_energy"]  = _num(row.get("energy"), 5)
        t["_rank"]    = rank_score(row)
        t["_print"]   = prints.get(k, {})
        tracks.append(t)
    attach_vectors(tracks)
    return tracks

def attach_vectors(tracks):
    """Turn each fingerprint into a unit vector, so similarity is one dot product."""
    labels = sorted({k for t in tracks for k in t["_print"]})
    index = {k: i for i, k in enumerate(labels)}
    for t in tracks:
        v = np.zeros(max(len(labels), 1), dtype=np.float32)
        for k, val in t["_print"].items():
            v[index[k]] = val
        n = np.linalg.norm(v)
        t["_vec"] = v / n if n else v

def rank_score(row):
    """Production quality plus enjoyment when present, else genre confidence."""
    q, e = _num(row.get("quality")), _num(row.get("enjoyment"))
    if q is not None and e is not None:
        score = (q + e) / 2
    else:
        score = _num(row.get("subgenre_1_pct"), 0) / 10
    if row.get("ending") == "cut off":
        score -= 1.0                  # most likely needs a re-render or an edit
    return round(score, 2)

# ---------------------------------------------------------------------------
# Compatibility
# ---------------------------------------------------------------------------
def _camelot(code):
    try:
        return int(code[:-1]), code[-1].upper()
    except (TypeError, ValueError, IndexError):
        return None

def keys_mix(a, b):
    """True when two Camelot codes sit next to each other on the wheel."""
    ca, cb = _camelot(a), _camelot(b)
    if not ca or not cb:
        return False
    (na, la), (nb, lb) = ca, cb
    step = min((na - nb) % 12, (nb - na) % 12)
    return (la == lb and step <= 1) or (la != lb and step == 0)

def bpm_close(a, b):
    if a is None or b is None:
        return True
    # Allow half and double time, e.g. 87 and 174
    return any(abs(a * m - b) / b <= BPM_TOLERANCE for m in (0.5, 1, 2))

# ---------------------------------------------------------------------------
# Listen-first list
# ---------------------------------------------------------------------------
def best_takes(tracks):
    """{idea: [takes best first]}."""
    ideas = {}
    for t in tracks:
        ideas.setdefault(t.get("idea") or t["file"], []).append(t)
    for takes in ideas.values():
        takes.sort(key=lambda t: -t["_rank"])
    return ideas

def write_listen_first(folder, ideas):
    rows = []
    firsts = sorted((takes[0] for takes in ideas.values()), key=lambda t: -t["_rank"])
    for i, t in enumerate(firsts, 1):
        rows.append(_listen_row(i, "best take", t, len(ideas[t.get("idea") or t["file"]])))
    rest = sorted((t for takes in ideas.values() for t in takes[1:]), key=lambda t: -t["_rank"])
    for i, t in enumerate(rest, len(rows) + 1):
        rows.append(_listen_row(i, "other take", t, len(ideas[t.get("idea") or t["file"]])))
    cols = ["order", "pick", "file", "folder", "idea", "takes_of_idea", "score", "lane", "artist",
            "subgenre_1", "bpm", "camelot", "energy", "ending", "quality", "enjoyment"]
    _write_csv(os.path.join(folder, LISTEN_NAME), cols, rows)
    return len(firsts)

def _listen_row(order, pick, t, n_takes):
    return {"order": order, "pick": pick, "file": t["file"], "folder": t.get("folder", ""),
            "idea": t.get("idea", ""), "takes_of_idea": n_takes, "score": t["_rank"],
            "lane": t.get("lane", ""), "artist": t.get("artist", ""), "subgenre_1": t.get("subgenre_1", ""),
            "bpm": t.get("bpm", ""), "camelot": t.get("camelot", ""), "energy": t.get("energy", ""),
            "ending": t.get("ending", ""), "quality": t.get("quality", ""), "enjoyment": t.get("enjoyment", "")}

# ---------------------------------------------------------------------------
# Releases
# ---------------------------------------------------------------------------
def group_releases(pool, size):
    """Greedy: seed with the best unused track, then add the tracks that fit
    best by sound, key and quality, within the release's BPM range."""
    pool = sorted(pool, key=lambda t: -t["_rank"])
    if not pool:
        return [], []
    vecs  = np.stack([t["_vec"] for t in pool])
    ranks = np.array([t["_rank"] for t in pool])
    used  = np.zeros(len(pool), dtype=bool)
    releases, leftover = [], []
    for seed in range(len(pool)):
        if used[seed]:
            continue
        used[seed] = True
        members = [seed]
        while len(members) < size:
            bpms = sorted(pool[m]["_bpm"] for m in members if pool[m]["_bpm"] is not None)
            mid = bpms[len(bpms) // 2] if bpms else None
            sims = vecs @ vecs[members].mean(axis=0)
            fit = sims * 2 + ranks / 10
            fit[used] = -np.inf
            pick = None
            # Best fits first; the key and BPM checks only run on the top few
            for i in np.argsort(-fit)[:200]:
                if not np.isfinite(fit[i]):
                    break
                if not bpm_close(pool[i]["_bpm"], mid):
                    continue
                mix = sum(keys_mix(pool[i].get("camelot"), pool[m].get("camelot")) for m in members)
                score = fit[i] + 0.5 * mix / len(members)
                if pick is None or score > pick[0]:
                    pick = (score, i)
            if pick is None:
                break
            used[pick[1]] = True
            members.append(pick[1])
        if len(members) >= min(MIN_RELEASE, size):
            releases.append([pool[m] for m in members])
        else:
            leftover.extend(pool[m] for m in members)
    return releases, leftover

def sequence(members):
    """Order a release: open mid energy, build to a peak about two thirds in,
    ease off at the end, and prefer key moves that mix."""
    n = len(members)
    if n <= 2:
        return sorted(members, key=lambda t: t["_energy"])
    lo = min(t["_energy"] for t in members)
    hi = max(t["_energy"] for t in members)
    def target(i):
        x = i / (n - 1)
        shape = math.sin(math.pi * min(x / 0.66, 1.0) / 2) if x <= 0.66 else 1 - (x - 0.66) / 0.34 * 0.5
        return lo + (hi - lo) * (0.4 + 0.6 * shape)
    left = sorted(members, key=lambda t: -t["_rank"])
    order = [min(left, key=lambda t: abs(t["_energy"] - target(0)) - t["_rank"] / 100)]
    left.remove(order[0])
    for i in range(1, n):
        prev = order[-1]
        nxt = min(left, key=lambda t: abs(t["_energy"] - target(i))
                  - (1.0 if keys_mix(prev.get("camelot"), t.get("camelot")) else 0))
        left.remove(nxt)
        order.append(nxt)
    return order

def write_releases(folder, ideas, size):
    by_lane = {}
    for takes in ideas.values():
        best = takes[0]
        by_lane.setdefault(best.get("lane") or "Unsorted", []).append(best)
    kind = "EP" if size <= 7 else "Album"
    rows, summary = [], []
    for lane in sorted(by_lane):
        releases, leftover = group_releases(by_lane[lane], size)
        for r, members in enumerate(releases, 1):
            name = f"{lane} - {kind} {r:02d}"
            ordered = sequence(members)
            bpms = [m["_bpm"] for m in ordered if m["_bpm"] is not None]
            for pos, t in enumerate(ordered, 1):
                prev = ordered[pos - 2] if pos > 1 else None
                rows.append({
                    "release": name, "position": pos, "file": t["file"], "folder": t.get("folder", ""),
                    "idea": t.get("idea", ""), "lane": lane, "artist": t.get("artist", ""),
                    "subgenre_1": t.get("subgenre_1", ""), "bpm": t.get("bpm", ""),
                    "camelot": t.get("camelot", ""), "energy": t.get("energy", ""), "score": t["_rank"],
                    "mixes_with_previous": "" if prev is None else ("yes" if keys_mix(prev.get("camelot"), t.get("camelot")) else "no"),
                    "release_bpm_range": f"{min(bpms):.0f}-{max(bpms):.0f}" if bpms else "",
                })
        summary.append((lane, len(releases), len(leftover)))
        for t in leftover:
            rows.append({"release": f"{lane} - unplaced", "position": "", "file": t["file"],
                         "folder": t.get("folder", ""), "idea": t.get("idea", ""), "lane": lane,
                         "artist": t.get("artist", ""), "subgenre_1": t.get("subgenre_1", ""),
                         "bpm": t.get("bpm", ""), "camelot": t.get("camelot", ""),
                         "energy": t.get("energy", ""), "score": t["_rank"],
                         "mixes_with_previous": "", "release_bpm_range": ""})
    cols = ["release", "position", "file", "folder", "idea", "lane", "artist", "subgenre_1", "bpm",
            "camelot", "energy", "score", "mixes_with_previous", "release_bpm_range"]
    _write_csv(os.path.join(folder, RELEASES_NAME), cols, rows)
    return summary, kind

def _write_csv(path, cols, rows):
    """Write with a BOM for Excel. If the file is open in Excel, write a dated copy instead."""
    try:
        target = path
        f = open(target, "w", encoding="utf-8-sig", newline="")
    except OSError:
        base, ext = os.path.splitext(path)
        target = f"{base}_{datetime.now().strftime('%Y-%m-%d_%H%M')}{ext}"
        f = open(target, "w", encoding="utf-8-sig", newline="")
    with f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return target

# ---------------------------------------------------------------------------
# Entry point for the web UI
# ---------------------------------------------------------------------------
def plan(folder, size, catalogue_name, fingerprint_name):
    """Write both files. Returns lines for the progress log."""
    folder = (folder or "").strip().strip('"')
    if not folder or not os.path.isdir(folder):
        return ["Folder not found. Enter a folder path or use Browse."]
    if not os.path.exists(os.path.join(folder, catalogue_name)):
        return [f"No {catalogue_name} in this folder yet. Run a batch first."]
    size = max(2, int(size or 6))
    tracks = read_tracks(folder, catalogue_name, fingerprint_name)
    if not tracks:
        return ["No analysed tracks in the catalogue yet."]
    if not any(t.get("bpm") for t in tracks):
        return [f"{catalogue_name} comes from an older SGD version. Run the batch again first."]
    ideas = best_takes(tracks)
    n_ideas = write_listen_first(folder, ideas)
    summary, kind = write_releases(folder, ideas, size)
    lines = [f"{len(tracks)} track(s), {n_ideas} idea(s).",
             f"Wrote {LISTEN_NAME}: the best take of each idea first, best at the top."]
    if not any(t.get("quality") for t in tracks):
        lines.append("No quality scores in the catalogue, so takes are ranked by genre confidence instead.")
    for lane, n, left in summary:
        lines.append(f"{lane}: {n} {kind}(s) of up to {size} tracks, {left} unplaced.")
    lines.append(f"Wrote {RELEASES_NAME}. Edit it in Excel as you like.")
    return lines
