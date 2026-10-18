# Copyright (c) 2026 by Haizy Tiles
#Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
Library report: one HTML page next to the catalogue that opens in any browser.

- Sound map: every track as a dot, placed so tracks that sound alike sit
  close together, coloured by lane. Hover for details, click to play.
- Overview: lanes, subgenres, BPM, keys on the Camelot wheel, energy,
  quality, vocals and moods.
- Release strips: each planned release as a row of tracks with energy and
  key moves.
"""

import csv
import html
import os
import urllib.parse
from collections import Counter
from datetime import datetime

import numpy as np
import plotly.graph_objects as go

import release_planner

REPORT_NAME  = "sgd_report.html"
MAX_STRIPS   = 300

SURFACE   = "#1a1a19"
PANEL     = "#232322"
TEXT      = "#f0efec"
TEXT_2    = "#c3c2b7"
MUTED     = "#8a8984"
GRID      = "#33332f"
ACCENT    = "#3987e5"
# Categorical slots in fixed order (dark mode steps), grey for Unsorted
LANE_COLOURS = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#008300"]
UNSORTED_COLOUR = "#6f6e69"
# One hue, light to dark reversed for a dark surface: low energy dim, high energy bright
ENERGY_SCALE = ["#1d3b5e", "#24518a", "#2a68b5", "#3987e5", "#6aa6ee", "#a3c8f5"]

# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------
def _layout(fig, title, height=320, **kw):
    fig.update_layout(
        title=dict(text=title, x=0, font=dict(size=15, color=TEXT)),
        paper_bgcolor=PANEL, plot_bgcolor=PANEL, height=height,
        font=dict(family="Segoe UI, system-ui, sans-serif", size=12, color=TEXT_2),
        margin=dict(l=10, r=10, t=44, b=10),
        hoverlabel=dict(bgcolor="#0f0f0e", bordercolor=GRID, font=dict(color=TEXT)),
        **{"showlegend": False, **kw})
    fig.update_xaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False, linecolor=GRID)
    return fig

def _hbar(counts, title, n=15, colour=ACCENT):
    items = counts.most_common(n)[::-1]
    fig = go.Figure(go.Bar(x=[v for _, v in items], y=[k for k, _ in items], orientation="h",
                           marker=dict(color=colour, cornerradius=4),
                           text=[v for _, v in items], textposition="outside",
                           textfont=dict(color=TEXT_2), cliponaxis=False,
                           hovertemplate="%{y}: %{x} track(s)<extra></extra>"))
    fig = _layout(fig, title, height=max(220, 26 * len(items) + 70),
                  xaxis=dict(showgrid=False, showticklabels=False,
                             range=[0, max((v for _, v in items), default=1) * 1.12]))
    fig.update_layout(margin=dict(r=40))
    return fig

def _hist(values, title, xtitle, bins=None):
    fig = go.Figure(go.Histogram(x=values, xbins=bins, marker=dict(color=ACCENT, line=dict(color=PANEL, width=2)),
                                 hovertemplate=f"{xtitle} %{{x}}: %{{y}} track(s)<extra></extra>"))
    return _layout(fig, title, xaxis_title=xtitle, yaxis_title="tracks", bargap=0.05)

def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# Sound map
# ---------------------------------------------------------------------------
def map_coords(tracks):
    """2D positions from each track's sound vector (genre scores plus CLAP).
    t-SNE keeps neighbours together; PCA is the fallback."""
    X = np.stack([t["_vec"] for t in tracks]).astype(np.float32)
    if len(tracks) < 5:
        return np.c_[np.arange(len(tracks)), np.zeros(len(tracks))]
    X = X - X.mean(axis=0)
    # PCA to at most 50 dimensions first, which also speeds up t-SNE
    k = min(50, X.shape[1], len(X) - 1)
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    P = U[:, :k] * S[:k]
    try:
        from sklearn.manifold import TSNE
        return TSNE(n_components=2, perplexity=min(30, max(2, len(X) // 4)), init="pca",
                    random_state=0).fit_transform(P)
    except Exception as e:
        print(f"t-SNE not available ({str(e)[:100]}), using PCA for the map.")
        return P[:, :2]

def sound_map(tracks, lanes):
    xy = map_coords(tracks)
    fig = go.Figure()
    for i, lane in enumerate(lanes):
        idx = [j for j, t in enumerate(tracks) if (t.get("lane") or "Unsorted") == lane]
        colour = UNSORTED_COLOUR if lane == "Unsorted" else LANE_COLOURS[i % len(LANE_COLOURS)]
        ts = [tracks[j] for j in idx]
        fig.add_trace(go.Scattergl(
            x=xy[idx, 0], y=xy[idx, 1], mode="markers", name=f"{lane} ({len(idx)})",
            marker=dict(size=8, color=colour, line=dict(color=PANEL, width=1)),
            customdata=[[_audio_src(t), t["file"]] for t in ts],
            text=[_hover(t) for t in ts], hovertemplate="%{text}<extra></extra>"))
    _layout(fig, "", height=680, showlegend=True,
            legend=dict(orientation="h", x=0, y=1.0, yanchor="bottom", font=dict(color=TEXT_2)))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, scaleanchor="x")
    return fig

def _hover(t):
    bits = [f"<b>{html.escape(t['file'])}</b>",
            f"{html.escape(t.get('lane') or 'Unsorted')} · {html.escape(t.get('subgenre_1', ''))}",
            f"{t.get('bpm', '')} BPM · {t.get('camelot', '')} · energy {t.get('energy', '')}"]
    extra = " · ".join(x for x in (t.get("vocals", ""), t.get("mood_1", "")) if x)
    if extra:
        bits.append(html.escape(extra))
    if t.get("quality"):
        bits.append(f"quality {t['quality']} · enjoyment {t.get('enjoyment', '')}")
    return "<br>".join(bits)

def _audio_src(t):
    rel = os.path.join(t.get("folder") or ".", t["file"]).replace("\\", "/")
    return urllib.parse.quote(rel)

# ---------------------------------------------------------------------------
# Overview charts
# ---------------------------------------------------------------------------
CAMELOT_ORDER = [f"{n}{l}" for n in range(1, 13) for l in ("A", "B")]

def camelot_wheel(tracks):
    counts = Counter(t.get("camelot") for t in tracks if t.get("camelot"))
    fig = go.Figure()
    for ring, letter in ((0, "A"), (1, "B")):
        codes = [f"{n}{letter}" for n in range(1, 13)]
        fig.add_trace(go.Barpolar(
            r=[1] * 12, base=[ring + 0.3] * 12, theta=[(n - 1) * 30 for n in range(1, 13)], width=[28] * 12,
            marker=dict(color=[counts.get(c, 0) for c in codes], colorscale=ENERGY_SCALE,
                        cmin=0, cmax=max(counts.values(), default=1), line=dict(color=PANEL, width=2)),
            text=codes, customdata=[counts.get(c, 0) for c in codes],
            hovertemplate="%{text}: %{customdata} track(s)<extra></extra>"))
    fig.update_layout(polar=dict(bgcolor=PANEL, radialaxis=dict(visible=False, range=[0, 2.3]),
                                 angularaxis=dict(direction="clockwise", rotation=90, tickmode="array",
                                                  tickvals=[(n - 1) * 30 for n in range(1, 13)],
                                                  ticktext=[str(n) for n in range(1, 13)],
                                                  gridcolor=GRID, linecolor=GRID, color=TEXT_2)))
    fig = _layout(fig, "Keys on the Camelot wheel (inner A = minor, outer B = major)", height=400)
    fig.update_layout(margin=dict(t=60, b=30))
    return fig

def overview(tracks):
    figs = []
    figs.append(_hbar(Counter(t.get("lane") or "Unsorted" for t in tracks), "Tracks per lane", n=10))
    figs.append(_hbar(Counter(t.get("subgenre_1") for t in tracks if t.get("subgenre_1")), "Top subgenres", n=15))
    bpms = [b for b in (_num(t.get("bpm")) for t in tracks) if b]
    figs.append(_hist(bpms, "BPM", "BPM", bins=dict(size=2)))
    figs.append(camelot_wheel(tracks))
    figs.append(_hist([_num(t.get("energy")) for t in tracks if _num(t.get("energy"))], "Energy", "energy (1-10)",
                      bins=dict(start=0.5, end=10.5, size=1)))
    q = [_num(t.get("quality")) for t in tracks if _num(t.get("quality")) is not None]
    if q:
        figs.append(_hist(q, "Production quality", "quality (1-10)", bins=dict(size=0.25)))
    lufs = [_num(t.get("lufs")) for t in tracks if _num(t.get("lufs")) is not None]
    if lufs:
        figs.append(_hist(lufs, "Loudness", "LUFS", bins=dict(size=1)))
    if any(t.get("vocals") for t in tracks):
        figs.append(_hbar(Counter(t["vocals"] for t in tracks if t.get("vocals")), "Vocals", n=6))
        moods = Counter(m for t in tracks for m in (t.get("mood_1"), t.get("mood_2")) if m)
        figs.append(_hbar(moods, "Moods (top two per track)", n=12))
    return figs

# ---------------------------------------------------------------------------
# Release strips (plain HTML, so thousands of releases stay light)
# ---------------------------------------------------------------------------
def release_strips(folder):
    path = os.path.join(folder, release_planner.RELEASES_NAME)
    if not os.path.exists(path):
        return "<p class='note'>No sgd_releases.csv yet. Press Plan releases, then build the report again.</p>", 0
    releases = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("position"):
                releases.setdefault(row["release"], []).append(row)
    out = []
    for name, rows in list(releases.items())[:MAX_STRIPS]:
        rows.sort(key=lambda r: int(r["position"]))
        cells = []
        for r in rows:
            e = _num(r.get("energy")) or 5
            shade = ENERGY_SCALE[min(len(ENERGY_SCALE) - 1, int((e - 1) / 9 * len(ENERGY_SCALE)))]
            join = ""
            if r.get("mixes_with_previous") == "no":
                join = "<span class='clash' title='Keys do not mix with the previous track'>✕</span>"
            elif r.get("mixes_with_previous") == "yes":
                join = "<span class='mix' title='Keys mix with the previous track'>→</span>"
            cells.append(
                f"{join}<div class='cell' style='--h:{int(e * 6)}px'"
                f" data-src='{_audio_src(r)}' title='{html.escape(r['file'], quote=True)}'>"
                f"<div class='bar' style='background:{shade}'></div>"
                f"<div class='name'>{html.escape(r.get('song') or r['file'])}</div>"
                f"<div class='meta'>{html.escape(r.get('bpm', ''))} · {html.escape(r.get('camelot', ''))} · e{html.escape(r.get('energy', ''))}</div></div>")
        rng = rows[0].get("release_bpm_range", "")
        out.append(f"<div class='strip'><div class='rname'>{html.escape(name)}<span>{len(rows)} tracks · {html.escape(rng)} BPM</span></div>"
                   f"<div class='cells'>{''.join(cells)}</div></div>")
    more = len(releases) - MAX_STRIPS
    if more > 0:
        out.append(f"<p class='note'>{more} more release(s) in sgd_releases.csv.</p>")
    return "\n".join(out), len(releases)

# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SGD Library report</title>
<style>
:root {{ --surface:{surface}; --panel:{panel}; --text:{text}; --text2:{text2}; --muted:{muted}; --grid:{grid}; --accent:{accent}; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--surface); color:var(--text); font-family:"Segoe UI",system-ui,sans-serif; }}
main {{ max-width:1400px; margin:0 auto; padding:16px 16px 120px; }}
h1 {{ font-size:22px; margin:8px 0 2px; }} h2 {{ font-size:17px; margin:28px 0 10px; }}
.sub {{ color:var(--text2); font-size:13px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:16px 0; }}
.tile {{ background:var(--panel); border-radius:8px; padding:12px 14px; }}
.tile b {{ display:block; font-size:24px; color:var(--text); }} .tile span {{ color:var(--text2); font-size:12px; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(420px,1fr)); gap:12px; }}
.card {{ background:var(--panel); border-radius:8px; overflow:hidden; }}
.note {{ color:var(--muted); font-size:13px; }}
#player {{ position:fixed; left:0; right:0; bottom:0; background:#0f0f0e; border-top:1px solid var(--grid);
          padding:10px 16px; display:flex; gap:14px; align-items:center; z-index:10; }}
#player audio {{ flex:1; max-width:700px; }} #now {{ color:var(--text2); font-size:13px; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; }}
.strip {{ background:var(--panel); border-radius:8px; padding:10px 12px; margin-bottom:8px; }}
.rname {{ font-size:13px; font-weight:600; margin-bottom:8px; }} .rname span {{ color:var(--muted); font-weight:400; margin-left:10px; }}
.cells {{ display:flex; align-items:flex-end; gap:4px; overflow-x:auto; }}
.cell {{ width:120px; flex:none; cursor:pointer; }}
.cell .bar {{ height:var(--h); min-height:4px; border-radius:4px 4px 0 0; margin-bottom:4px; }}
.cell:hover .name {{ color:var(--accent); }}
.cell .name {{ font-size:12px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
.cell .meta {{ font-size:11px; color:var(--muted); }}
.mix, .clash {{ align-self:center; font-size:13px; padding:0 2px; color:var(--muted); }} .clash {{ color:#e66767; }}
@media (max-width:600px) {{ .grid {{ grid-template-columns:1fr; }} }}
</style></head>
<body><main>
<h1>Library report</h1>
<div class="sub">{folder} · built {built}</div>
<div class="tiles">{tiles}</div>
<h2>Sound map</h2>
<p class="note">Tracks that sound alike sit close together. Click a lane in the key to hide it, double-click to show only that lane. Click a dot to play the track.</p>
<div class="card">{map}</div>
<h2>Overview</h2>
<div class="grid">{charts}</div>
<h2>Release strips</h2>
<p class="note">Each row is one planned release in running order. Bar height is energy. → means the keys mix, ✕ means they clash. Click a track to play it.</p>
{strips}
</main>
<div id="player"><audio id="audio" controls></audio><div id="now">Click a dot or a track to play it.</div></div>
<script>
function play(src, name) {{
  const a = document.getElementById('audio');
  a.src = src; a.play();
  document.getElementById('now').textContent = name;
}}
document.querySelectorAll('.cell').forEach(c => c.addEventListener('click', () => play(c.dataset.src, c.title)));
const map = document.getElementById('soundmap');
if (map) map.on('plotly_click', e => {{ const d = e.points[0].customdata; play(d[0], d[1]); }});
</script>
</body></html>
"""

def write_report(folder, catalogue_name, fingerprint_name):
    """Build sgd_report.html. Returns lines for the progress log."""
    folder = (folder or "").strip().strip('"')
    if not folder or not os.path.isdir(folder):
        return ["Folder not found. Enter a folder path or use Browse."]
    if not os.path.exists(os.path.join(folder, catalogue_name)):
        return [f"No {catalogue_name} in this folder yet. Run a batch first."]
    tracks = release_planner.read_tracks(folder, catalogue_name, fingerprint_name)
    if not tracks:
        return ["No analysed tracks in the catalogue yet."]
    lane_counts = Counter(t.get("lane") or "Unsorted" for t in tracks)
    lanes = sorted(l for l in lane_counts if l != "Unsorted") + (["Unsorted"] if "Unsorted" in lane_counts else [])
    map_fig = sound_map(tracks, lanes)
    charts = overview(tracks)
    strips, n_releases = release_strips(folder)
    songs = len({t.get("idea") or t["file"] for t in tracks})
    q = [_num(t.get("quality")) for t in tracks if _num(t.get("quality")) is not None]
    tiles = [("tracks", len(tracks)), ("songs", songs), ("lanes", len(lanes)), ("releases planned", n_releases)]
    if q:
        tiles.append(("median quality", f"{np.median(q):.1f}"))
    page = PAGE.format(
        surface=SURFACE, panel=PANEL, text=TEXT, text2=TEXT_2, muted=MUTED, grid=GRID, accent=ACCENT,
        folder=html.escape(folder), built=datetime.now().strftime("%Y-%m-%d %H:%M"),
        tiles="".join(f"<div class='tile'><b>{v}</b><span>{k}</span></div>" for k, v in tiles),
        map=map_fig.to_html(full_html=False, include_plotlyjs=True, div_id="soundmap",
                            config={"displaylogo": False}),
        charts="".join(f"<div class='card'>{f.to_html(full_html=False, include_plotlyjs=False, config={'displaylogo': False})}</div>"
                       for f in charts),
        strips=strips)
    path = os.path.join(folder, REPORT_NAME)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
    except OSError as e:
        return [f"Could not write {REPORT_NAME}: {e}"]
    return [f"Wrote {REPORT_NAME} with {len(tracks)} track(s). It should open in your browser."]
