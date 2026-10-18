# Copyright (c) 2026 by Haizy Tiles
#Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Vocals, mood and instrument tags from LAION's CLAP music model, plus text
search ("warm Rhodes deep house") over a finished batch.

CLAP turns a clip of audio and a line of text into points in the same space.
A tag scores high when its line in clap_tags.csv sits close to the track.
Edit clap_tags.csv to add or change tags; it is read again when it changes.
"""

import csv
import json
import os

import numpy as np

MODEL_ID      = "laion/larger_clap_music"
SR            = 48000
CLIP_SECONDS  = 10
MAX_CLIPS     = 6
EMBED_NAME    = "sgd_clap.jsonl"
TAGS_FILE     = os.path.join(os.path.dirname(os.path.abspath(__file__)), "clap_tags.csv")
COLUMNS       = ["vocals", "vocals_pct", "mood_1", "mood_2", "instruments"]

_model = _processor = _device = None
_unavailable = None
_scale = 30.0
_tags = None
_tags_mtime = None

def available():
    return _load() is not None

def unavailable_reason():
    return _unavailable

def _load():
    global _model, _processor, _device, _unavailable, _scale
    if _model is not None or _unavailable is not None:
        return _model
    try:
        print("Loading CLAP model...")
        import torch
        from transformers import ClapModel, ClapProcessor
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _processor = ClapProcessor.from_pretrained(MODEL_ID)
        _model = ClapModel.from_pretrained(MODEL_ID).to(_device).eval()
        if hasattr(_model, "logit_scale_a"):
            _scale = float(_model.logit_scale_a.detach().exp().clamp(max=100))
        print(f"CLAP ready on {_device}.")
    except Exception as e:
        _model = None
        _unavailable = str(e)[:200]
        print(f"Vocal, mood and instrument tags are off: {_unavailable}")
    return _model

def _vectors(out):
    """get_*_features returns a tensor in most transformers versions and an
    output object in some. Return unit vectors as numpy."""
    if not hasattr(out, "detach"):
        out = getattr(out, "pooler_output", None) if getattr(out, "pooler_output", None) is not None else out[0]
    v = out.detach().float().cpu().numpy()
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-9)

def embed_text(texts):
    import torch
    tok = _processor.tokenizer(list(texts), padding=True, return_tensors="pt").to(_device)
    with torch.no_grad():
        return _vectors(_model.get_text_features(**tok))

def embed_audio(y48):
    """One unit vector for a mono 48 kHz signal: the mean of up to MAX_CLIPS
    10 s clips spread over the middle 90% of the track."""
    import torch
    clip = SR * CLIP_SECONDS
    edge = int(len(y48) * 0.05)
    body = y48[edge:len(y48) - edge] if len(y48) - 2 * edge >= clip else y48
    if len(body) <= clip:
        clips = [np.pad(body, (0, clip - len(body)))]
    else:
        n = min(MAX_CLIPS, max(1, len(body) // clip))
        clips = [body[s:s + clip] for s in np.linspace(0, len(body) - clip, n, dtype=int)]
    feats = _processor.feature_extractor([c.astype(np.float32) for c in clips],
                                         sampling_rate=SR, return_tensors="pt")
    feats = {k: v.to(_device) for k, v in feats.items()}
    with torch.no_grad():
        v = _vectors(_model.get_audio_features(**feats)).mean(axis=0)
    return v / max(np.linalg.norm(v), 1e-9)

# ---------------------------------------------------------------------------
# Tags
# ---------------------------------------------------------------------------
def load_tags():
    """{group: (pick, [tag names], text vectors)} from clap_tags.csv."""
    global _tags, _tags_mtime
    try:
        mtime = os.path.getmtime(TAGS_FILE)
    except OSError:
        return {}
    if _tags is not None and mtime == _tags_mtime:
        return _tags
    groups = {}
    with open(TAGS_FILE, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            g, tag, prompt = (row.get(k, "").strip() for k in ("group", "tag", "prompt"))
            if g and tag and prompt:
                pick = int(row.get("pick") or 1)
                groups.setdefault(g, [pick, [], []])
                groups[g][1].append(tag)
                groups[g][2].append(prompt)
    _tags = {g: (pick, names, embed_text(prompts)) for g, (pick, names, prompts) in groups.items()}
    _tags_mtime = mtime
    return _tags

def tag(emb):
    """Catalogue columns for one track's CLAP vector."""
    row = {}
    for group, (pick, names, vecs) in load_tags().items():
        logits = vecs @ emb * _scale
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        order = np.argsort(-probs)
        if group == "vocals":
            row["vocals"], row["vocals_pct"] = names[order[0]], round(float(probs[order[0]]) * 100)
        elif group == "mood":
            for i, k in enumerate(order[:2], 1):
                row[f"mood_{i}"] = names[k]
        else:
            # Several can be true at once: keep the clear ones, up to pick
            floor = max(0.10, 1.5 / len(names))
            keep = [names[k] for k in order[:pick] if probs[k] >= floor] or [names[order[0]]]
            row[group] = ", ".join(keep)
    return row

def analyse(y48):
    """(columns, vector) for a mono 48 kHz signal, or ({}, None) if CLAP is off
    or fails on this track. A failure never fails the whole track."""
    if _load() is None:
        return {}, None
    try:
        emb = embed_audio(y48)
        return tag(emb), emb
    except Exception as e:
        print(f"CLAP tags failed for this track: {str(e)[:200]}")
        return {}, None

def save_embedding(root, row, emb):
    try:
        with open(os.path.join(root, EMBED_NAME), "a", encoding="utf-8") as f:
            f.write(json.dumps({"file": row["file"], "folder": row["folder"], "model": MODEL_ID,
                                "vec": [round(float(x), 4) for x in emb]}, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"Could not save CLAP vector for {row['file']}: {e}")

def read_embeddings(folder):
    """{track key: (folder, file, unit vector)}. The last line for a track wins."""
    out = {}
    path = os.path.join(folder, EMBED_NAME)
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except ValueError:
                continue
            key = os.path.normcase(os.path.normpath(os.path.join(d.get("folder") or ".", d.get("file", ""))))
            out[key] = (d.get("folder") or ".", d.get("file", ""), np.asarray(d.get("vec", []), dtype=np.float32))
    return out

# ---------------------------------------------------------------------------
# Text search
# ---------------------------------------------------------------------------
def search(folder, text, n=25):
    """[(similarity, folder, file)] for the tracks closest to the text."""
    embs = read_embeddings(folder)
    if not embs:
        raise ValueError(f"No {EMBED_NAME} in this folder. Run a batch with tags on first.")
    if _load() is None:
        raise ValueError(f"CLAP is not available: {_unavailable}")
    q = embed_text([text])[0]
    items = list(embs.values())
    sims = np.stack([v for _, _, v in items]) @ q
    best = np.argsort(-sims)[:n]
    return [(round(float(sims[i]), 3), items[i][0], items[i][1]) for i in best]
