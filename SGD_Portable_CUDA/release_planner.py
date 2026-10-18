# Copyright (c) 2026 by Haizy Tiles
#Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Release planning from a finished batch catalogue.

- sgd_listen_first.csv: the best take of every song, best first, then the
  other takes. Work down it from the top, or use the Listen tab.
- sgd_releases.csv: suggested EPs or albums, one lane at a time. Tracks in
  a release sound alike, sit in a similar BPM range, mix well by key, never
  share a song, and are put in an order with a rise and fall in energy.
- sgd_same_audio.csv: pairs of tracks that share the same audio.

A song is every take with the same title once take numbers and tags are
removed, so ACID CELLULOID [E] 001 and 004 are one song. Keep, Maybe and Bin from the Listen tab are saved in
sgd_choices.csv and change the ranking here.

All files are plain suggestions to edit in Excel.
"""

import csv
import json
import math
import os
from datetime import datetime

import numpy as np

import audio_id
import clap_tags

LISTEN_NAME   = "sgd_listen_first.csv"
RELEASES_NAME = "sgd_releases.csv"
SAME_NAME     = "sgd_same_audio.csv"
CHOICES_NAME  = "sgd_choices.csv"
CHOICE_BONUS  = {"keep": 3.0, "maybe": 1.0}
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

def read_choices(folder):
    """{track key: keep, maybe or bin}. The last choice for a track wins."""
    out = {}
    path = os.path.join(folder, CHOICES_NAME)
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("file"):
                out[_key(row.get("folder"), row["file"])] = (row.get("choice") or "").strip().lower()
    return out

def save_choice(folder, file, sub, choice):
    """Append one choice. Returns False if the file is locked (open in Excel)."""
    path = os.path.join(folder, CHOICES_NAME)
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    try:
        with open(path, "a", encoding="utf-8-sig" if new else "utf-8", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["file", "folder", "choice", "at"])
            w.writerow([file, sub, choice, datetime.now().isoformat(timespec="seconds")])
        return True
    except OSError:
        return False

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
    claps = clap_tags.read_embeddings(folder)
    choices = read_choices(folder)
    tracks = []
    for k, row in rows.items():
        if row.get("status") != "ok":
            continue
        t = dict(row)
        t["_key"]     = k
        t["_bpm"]     = _num(row.get("bpm"))
        t["_energy"]  = _num(row.get("energy"), 5)
        t["_choice"]  = choices.get(k, "")
        t["_rank"]    = round(rank_score(row) + CHOICE_BONUS.get(t["_choice"], 0), 2)
        t["_print"]   = prints.get(k, {})
        t["_clap"]    = claps[k][2] if k in claps else None
        tracks.append(t)
    attach_vectors(tracks)
    return tracks

def attach_vectors(tracks):
    """Turn each fingerprint into a unit vector, so similarity is one dot product.
    When CLAP vectors are there, they count half: genre scores say what style a
    track is, CLAP says more about how it sounds."""
    labels = sorted({k for t in tracks for k in t["_print"]})
    index = {k: i for i, k in enumerate(labels)}
    clap_dim = max((len(t["_clap"]) for t in tracks if t["_clap"] is not None), default=0)
    for t in tracks:
        v = np.zeros(max(len(labels), 1), dtype=np.float32)
        for k, val in t["_print"].items():
            v[index[k]] = val
        n = np.linalg.norm(v)
        v = v / n if n else v
        if clap_dim:
            c = t["_clap"] if t["_clap"] is not None and len(t["_clap"]) == clap_dim else np.zeros(clap_dim, np.float32)
            v = np.concatenate([v, c]) / np.sqrt(2)
        t["_vec"] = v

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
def group_songs(tracks, same_audio):
    """Takes with the same title (idea name) are one song, so
    ACID CELLULOID [E] 001 and 004 never share a release. Shared audio is
    only noted, not used for grouping."""
    index = {t["_key"]: t for t in tracks}
    for t in tracks:
        t["song"] = t.get("idea") or t["file"]
        t["_same_audio"] = []
    for a, b, _, _ in same_audio:
        if a in index and b in index:
            index[a]["_same_audio"].append(index[b]["file"])
            index[b]["_same_audio"].append(index[a]["file"])

def best_takes(tracks):
    """{song: [takes best first]}. Binned takes go last."""
    songs = {}
    for t in tracks:
        songs.setdefault(t.get("song") or t.get("idea") or t["file"], []).append(t)
    for takes in songs.values():
        takes.sort(key=lambda t: (t["_choice"] == "bin", -t["_rank"]))
    return songs

def write_listen_first(folder, songs):
    firsts = [takes[0] for takes in songs.values() if takes[0]["_choice"] != "bin"]
    rest   = [t for takes in songs.values() for t in takes if t["_choice"] != "bin" and t is not takes[0]]
    binned = [t for takes in songs.values() for t in takes if t["_choice"] == "bin"]
    rows = []
    for pick, group in (("best take", firsts), ("other take", rest), ("binned", binned)):
        for t in sorted(group, key=lambda t: -t["_rank"]):
            rows.append(_listen_row(len(rows) + 1, pick, t, len(songs[t["song"]])))
    _write_csv(os.path.join(folder, LISTEN_NAME), LISTEN_COLUMNS, rows)
    return len(firsts)

LISTEN_COLUMNS = ["order", "pick", "choice", "file", "folder", "song", "idea", "takes_of_song",
                  "same_audio_as", "score", "lane", "artist", "subgenre_1", "bpm", "camelot", "energy",
                  "ending", "lufs", "vocals", "mood_1", "instruments", "quality", "enjoyment"]

def _listen_row(order, pick, t, n_takes):
    row = {c: t.get(c, "") for c in LISTEN_COLUMNS}
    row.update({"order": order, "pick": pick, "choice": t["_choice"], "takes_of_song": n_takes,
                "same_audio_as": "; ".join(t["_same_audio"]), "score": t["_rank"]})
    return row

def write_same_audio(folder, tracks, same_audio):
    by_key = {t["_key"]: t for t in tracks}
    rows = []
    for a, b, match, secs in sorted(same_audio, key=lambda x: -x[2]):
        ta, tb = by_key.get(a), by_key.get(b)
        if ta and tb:
            rows.append({"file_a": ta["file"], "folder_a": ta.get("folder", ""),
                         "file_b": tb["file"], "folder_b": tb.get("folder", ""),
                         "match_pct": round(match * 100), "seconds_shared": secs})
    _write_csv(os.path.join(folder, SAME_NAME),
               ["file_a", "folder_a", "file_b", "folder_b", "match_pct", "seconds_shared"], rows)
    return len(rows)

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

def write_releases(folder, songs, size):
    by_lane = {}
    for takes in songs.values():
        best = takes[0]
        if best["_choice"] != "bin":
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
                    "song": t.get("song", ""), "choice": t["_choice"], "vocals": t.get("vocals", ""),
                    "lane": lane, "artist": t.get("artist", ""),
                    "subgenre_1": t.get("subgenre_1", ""), "bpm": t.get("bpm", ""),
                    "camelot": t.get("camelot", ""), "energy": t.get("energy", ""), "score": t["_rank"],
                    "mixes_with_previous": "" if prev is None else ("yes" if keys_mix(prev.get("camelot"), t.get("camelot")) else "no"),
                    "release_bpm_range": f"{min(bpms):.0f}-{max(bpms):.0f}" if bpms else "",
                })
        summary.append((lane, len(releases), len(leftover)))
        for t in leftover:
            rows.append({"release": f"{lane} - unplaced", "position": "", "file": t["file"],
                         "folder": t.get("folder", ""), "song": t.get("song", ""), "choice": t["_choice"],
                         "vocals": t.get("vocals", ""), "lane": lane,
                         "artist": t.get("artist", ""), "subgenre_1": t.get("subgenre_1", ""),
                         "bpm": t.get("bpm", ""), "camelot": t.get("camelot", ""),
                         "energy": t.get("energy", ""), "score": t["_rank"],
                         "mixes_with_previous": "", "release_bpm_range": ""})
    cols = ["release", "position", "file", "folder", "song", "choice", "vocals", "lane", "artist",
            "subgenre_1", "bpm", "camelot", "energy", "score", "mixes_with_previous", "release_bpm_range"]
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
    same = audio_id.find_same_audio(audio_id.read_prints(folder))
    group_songs(tracks, same)
    songs = best_takes(tracks)
    n_same = write_same_audio(folder, tracks, same)
    n_songs = write_listen_first(folder, songs)
    summary, kind = write_releases(folder, songs, size)
    binned = sum(t["_choice"] == "bin" for t in tracks)
    lines = [f"{len(tracks)} track(s), {len(songs)} song(s)"
             + (f", {binned} binned" if binned else "") + ".",
             f"Wrote {LISTEN_NAME}: the best take of each song first, best at the top."]
    if os.path.exists(os.path.join(folder, audio_id.PRINT_NAME)):
        lines.append(f"{n_same} pair(s) of tracks share audio. See {SAME_NAME}.")
    if not any(t.get("quality") for t in tracks):
        lines.append("No quality scores in the catalogue, so takes are ranked by genre confidence instead.")
    for lane, n, left in summary:
        lines.append(f"{lane}: {n} {kind}(s) of up to {size} tracks, {left} unplaced.")
    lines.append(f"Wrote {RELEASES_NAME}. Edit it in Excel as you like.")
    return lines
