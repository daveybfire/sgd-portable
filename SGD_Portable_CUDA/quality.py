# Copyright (c) 2026 by Haizy Tiles
#Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Quality scores from Meta's Audiobox Aesthetics model (CC BY 4.0):
production quality, content enjoyment, production complexity and content
usefulness, each roughly on a 1-10 scale. Used to rank takes so the best
ones get listened to first. It rates sound, not taste.
"""

import numpy as np

SR = 16000
CHUNK_SECONDS = 60          # scored a minute at a time to keep memory low

_predictor = None
_unavailable = None         # reason, once loading has failed

COLUMNS = {"PQ": "quality", "CE": "enjoyment", "PC": "complexity", "CU": "usefulness"}

def available():
    return _load() is not None

def unavailable_reason():
    return _unavailable

def _load():
    global _predictor, _unavailable
    if _predictor is not None or _unavailable is not None:
        return _predictor
    try:
        print("Loading Audiobox Aesthetics model...")
        from audiobox_aesthetics.infer import initialize_predictor
        _predictor = initialize_predictor()
        print("Audiobox Aesthetics ready.")
    except Exception as e:
        _unavailable = str(e)[:200]
        print(f"Quality scores are off: {_unavailable}")
    return _predictor

def score(y):
    """{column: score} for a mono signal at 16 kHz, or {} if the model is not
    available or fails on this track. A failure never fails the whole track."""
    predictor = _load()
    if predictor is None:
        return {}
    try:
        return _score(predictor, y)
    except Exception as e:
        print(f"Quality score failed for this track: {str(e)[:200]}")
        return {}

def _score(predictor, y):
    import torch
    # Skip the first and last 5%, like the genre clips
    edge = int(len(y) * 0.05)
    body = y[edge:len(y) - edge] if len(y) - 2 * edge >= SR * 10 else y
    chunk = SR * CHUNK_SECONDS
    totals, weight = {}, 0.0
    for start in range(0, len(body), chunk):
        piece = body[start:start + chunk]
        if len(piece) < SR * 5 and weight > 0:
            break
        wav = torch.from_numpy(np.ascontiguousarray(piece, dtype=np.float32)).unsqueeze(0)
        result = predictor.forward([{"path": wav, "sample_rate": SR}])[0]
        w = len(piece)
        for axis, value in result.items():
            totals[axis] = totals.get(axis, 0.0) + value * w
        weight += w
    return {COLUMNS[a]: round(v / weight, 2) for a, v in totals.items() if a in COLUMNS}
