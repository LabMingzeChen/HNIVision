#!/usr/bin/env python3
"""Generate assets/gallery/01_methods.html from raw method outputs."""

import json
from pathlib import Path

GALLERY = Path("assets/gallery")

COLORS = {
    1: "#2D7D4F",  # Labeling - emerald
    2: "#4A6B3A",  # Classification - sage
    3: "#2563EB",  # Detection - blue
    4: "#B45309",  # Segmentation - amber
    5: "#B91C1C",  # VLM - red
}

SEG_COLORS = {
    "sky": "#b3d9f0", "tree": "#4b7d50", "grass": "#a0c364",
    "person": "#9e89c8", "wall": "#dcc3c3", "bannister": "#d7c8af",
    "sidewalk": "#afafaf", "pavement": "#afafaf", "fence": "#c8aaaa",
    "path": "#b9aa91", "stairs": "#a09682", "field": "#b4c882",
    "ground": "#9b8264", "water": "#82aad2", "building": "#c8c3be",
    "plant": "#82af6e", "earth": "#9b8264",
}

def seg_color(name):
    first = name.split(",")[0].strip().lower()
    return SEG_COLORS.get(first, "#a0a0a0")

def short_name(name):
    return name.split(",")[0].strip()

# --- Load data ---
with open(GALLERY / "01_methods_raw_outputs.json") as f:
    raw = json.load(f)

# --- Helpers ---
def bar(label, value, color, value_str=None, max_value=1.0):
    pct = min(100, (value / max_value) * 100)
    val = value_str if value_str is not None else f"{value:.2f}"
    return (
        f'<div class="bar-row">'
        f'<span class="bar-label">{label}</span>'
        f'<div class="bar-track"><div class="bar-fill" style="width:{pct:.1f}%;background:{color};"></div></div>'
        f'<span class="bar-value">{val}</span>'
        f'</div>'
    )

def hni_levels_html(coverage):
    """coverage: list of 4 'full'/'partial'/'empty'."""
    letters = ["H", "N", "A", "M"]
    pills = [f'<span class="hni-pill {s}">{l}</span>' for l, s in zip(letters, coverage)]
    return '<div class="hni-levels">' + "".join(pills) + '</div>'

def safe_get(d, key, default=""):
    return d.get(key, default) if isinstance(d, dict) else default

def safe_list(d, key="tags"):
    if isinstance(d, dict):
        v = d.get(key, [])
        return v if isinstance(v, list) else ([v] if v else [])
    return []

# --- Card 1: Labeling ---
labels_sorted = sorted(raw["labeling"]["labels"], key=lambda l: l["confidence"], reverse=True)
labels_top = labels_sorted[:8]
card1_bars = "".join([bar(l["description"], l["confidence"], COLORS[1]) for l in labels_top])

# --- Card 2: Classification ---
clf = raw["classification"]
top_class = clf["predicted_class"]
top_prob = clf["confidence"]
all_probs = clf["all_probs"]
other_classes = sorted([(k, v) for k, v in all_probs.items() if k != top_class and v >= 0.005],
                       key=lambda x: x[1], reverse=True)
card2_others = "".join([bar(name, prob, COLORS[2]) for name, prob in other_classes])

# --- Card 3: Detection ---
boxes = raw["detection"]["boxes"]
card3_boxes = ""
for box in boxes:
    coords = f"[{box['x1']:.0f}, {box['y1']:.0f}, {box['x2']:.0f}, {box['y2']:.0f}]"
    card3_boxes += f'''
    <div class="det-box">
      <div class="det-box-row">
        <span class="det-box-name">{box['class_name']}</span>
        <span class="det-box-conf">{box['confidence']:.2f}</span>
      </div>
      <div class="det-box-coords">bbox {coords}</div>
    </div>'''

total_dets = len(boxes)
avg_conf = sum(b["confidence"] for b in boxes) / max(1, len(boxes))

# --- Card 4: Segmentation ---
seg_shares = raw["segmentation"]["pixel_shares"]["shares"]
n_classes = raw["segmentation"]["n_classes_detected"]
seg_sorted = sorted(seg_shares.items(), key=lambda x: x[1], reverse=True)[:8]
card4_bars = "".join([
    bar(short_name(name), share, seg_color(name), f"{share*100:.1f}%", max_value=0.55)
    for name, share in seg_sorted
])

# --- Card 5: VLM ---
vlm_parsed = raw["vlm"]["parsed"]
vlm_summary = safe_get(vlm_parsed, "image_level_summary", "")
tokens = raw["vlm"].get("tokens_used", 0)

hum = vlm_parsed.get("human_presence", {})
nat = vlm_parsed.get("nature_detection", {})
act = vlm_parsed.get("activity_evidence", {})
mng = vlm_parsed.get("cultural_experiential_meaning", {})
hni_overall = vlm_parsed.get("hni_overall_interpretation", {})

# Human: presence_type + estimated_people_count
hum_types = hum.get("presence_type", []) if isinstance(hum, dict) else []
hum_count = hum.get("estimated_people_count", "") if isinstance(hum, dict) else ""
hum_main = hum_types[0] if hum_types else "—"
hum_display = hum_main + (f" · {hum_count}" if hum_count else "")

def dom_plus(d, list_key, dom_key):
    if not isinstance(d, dict):
        return "—"
    tags = d.get(list_key, []) or []
    dom = d.get(dom_key, "") or (tags[0] if tags else "")
    others = [t for t in tags if t != dom]
    out = dom if dom else "—"
    if others:
        out += f' <span class="extra">+{len(others)} more</span>'
    return out

nat_display = dom_plus(nat, "nature_types", "dominant_nature_type")
act_display = dom_plus(act, "activity_types", "dominant_activity")
mng_display = dom_plus(mng, "meaning_types", "dominant_meaning")

hni_present = hni_overall.get("is_hni_present", "—") if isinstance(hni_overall, dict) else "—"
hni_strength = hni_overall.get("hni_strength", "—") if isinstance(hni_overall, dict) else "—"
confidence = vlm_parsed.get("confidence", "—")

# --- HNI coverage per method ---
COVERAGE = {
    1: ["partial", "partial", "partial", "partial"],
    2: ["empty",   "partial", "partial", "partial"],
    3: ["full",    "empty",   "empty",   "empty"],
    4: ["full",    "full",    "empty",   "empty"],
    5: ["full",    "full",    "full",    "full"],
}

# --- CSS ---
CSS = """
:root { --bg:#fafafa; --card-bg:#fff; --border:#e5e7eb; --t1:#1f2937; --t2:#6b7280; --t3:#9ca3af; }
* { box-sizing: border-box; }
body { margin:0; padding:0; background:var(--bg);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  color: var(--t1); -webkit-font-smoothing: antialiased; }
.gallery { width:1500px; padding:40px; margin:0 auto; }
.gallery-header { margin-bottom:28px; }
.gallery-header h1 { margin:0 0 6px; font-size:26px; font-weight:700; letter-spacing:-0.5px; color:#14532d; }
.gallery-header p { margin:0; font-size:14px; color:var(--t2); }
.cards-row { display:grid; grid-template-columns:repeat(5,1fr); gap:16px; }
.method-card { background:var(--card-bg); border-radius:12px; border:1px solid var(--border);
  overflow:hidden; display:flex; flex-direction:column; box-shadow:0 1px 3px rgba(0,0,0,0.04); }
.card-header { background:var(--card-color); color:#fff; padding:14px 18px 16px; position:relative; }
.card-index { position:absolute; top:8px; right:14px; font-size:11px; opacity:0.7; font-weight:500; }
.card-title { font-size:18px; font-weight:700; margin:0; }
.card-body { padding:14px 18px 16px; flex:1; display:flex; flex-direction:column; }
.card-backend { font-size:11px; color:var(--card-color); font-weight:600; margin-bottom:10px; opacity:0.95; }
.card-image { width:100%; border-radius:4px; margin-bottom:6px; display:block; }
.card-image-caption { font-size:10px; color:var(--t3); margin-bottom:12px; }
.output-label { font-size:10px; font-weight:700; color:var(--t3); letter-spacing:0.6px; margin-bottom:10px; }
.bar-row { display:grid; grid-template-columns:1fr 75px 42px; gap:6px; align-items:center; margin-bottom:5px; }
.bar-label { font-size:12px; color:var(--t1); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-track { background:#f3f4f6; border-radius:999px; height:7px; overflow:hidden; }
.bar-fill { height:100%; border-radius:999px; }
.bar-value { font-size:10px; color:var(--t2); text-align:right; font-variant-numeric:tabular-nums; }
.top-class-card { background:rgba(74,107,58,0.08); border-radius:8px; padding:11px 13px; margin-bottom:12px;
  border-left:3px solid var(--card-color); }
.top-class-name { font-size:14px; font-weight:700; color:var(--card-color); line-height:1.2; }
.top-class-prob { font-size:11px; color:var(--card-color); margin-top:3px;
  font-family:"SF Mono",ui-monospace,monospace; }
.other-classes-h { font-size:9px; font-weight:700; color:var(--t3); margin:8px 0 5px;
  letter-spacing:0.6px; text-transform:uppercase; }
.det-box { background:rgba(37,99,235,0.07); padding:8px 10px; margin-bottom:5px; border-radius:5px; }
.det-box-row { display:flex; justify-content:space-between; align-items:baseline; }
.det-box-name { font-size:13px; font-weight:700; color:var(--card-color); }
.det-box-conf { font-size:12px; color:var(--card-color); font-weight:600; font-variant-numeric:tabular-nums; }
.det-box-coords { font-size:9px; color:var(--t2); font-family:"SF Mono",ui-monospace,monospace; margin-top:2px; }
.det-stats { margin-top:10px; padding-top:10px; border-top:1px solid var(--border); }
.det-stat-row { display:flex; justify-content:space-between; font-size:11px; margin-bottom:3px; color:var(--t1); }
.det-stat-val { font-weight:700; color:var(--card-color); font-variant-numeric:tabular-nums; }
.scene-summary { font-size:11px; color:var(--t1); line-height:1.55; margin-bottom:14px; }
.hni-tag { margin-bottom:7px; border-left:3px solid; padding-left:8px; }
.hni-tag.h { border-color:#c95960; }
.hni-tag.n { border-color:#3b6a3b; }
.hni-tag.a { border-color:#b07c2e; }
.hni-tag.m { border-color:#5e6dca; }
.hni-tag-label { font-size:9px; font-weight:700; color:var(--t3); letter-spacing:0.3px; text-transform:uppercase; }
.hni-tag-value { font-size:12px; color:var(--t1); margin-top:1px; }
.hni-tag-value .extra { color:var(--t3); font-size:10px; }
.vlm-meta { font-size:10px; color:var(--t3); margin-top:10px; padding-top:8px; border-top:1px solid var(--border);
  font-family:"SF Mono",ui-monospace,monospace; }
.card-footer { padding:11px 18px 13px; border-top:1px solid var(--border); background:#fafafa; }
.hni-levels-label { font-size:8px; font-weight:700; color:var(--t3); letter-spacing:0.6px; margin-bottom:6px;
  text-transform:uppercase; }
.hni-levels { display:flex; gap:6px; }
.hni-pill { width:22px; height:22px; border-radius:50%; font-size:10px; font-weight:700;
  display:inline-flex; align-items:center; justify-content:center; }
.hni-pill.full { background:var(--card-color); color:#fff; }
.hni-pill.partial { background:var(--card-color); opacity:0.4; color:#fff; }
.hni-pill.empty { background:transparent; border:1.5px solid #d1d5db; color:#9ca3af; }
.gallery-footer { margin-top:24px; padding-top:14px; border-top:2px solid #14532d;
  display:flex; justify-content:space-between; align-items:center; font-size:11px; color:var(--t2); }
.legend-pills { display:flex; gap:10px; align-items:center; }
.legend-pills .lp { width:11px; height:11px; border-radius:50%; display:inline-block; margin-right:4px; vertical-align:middle; }
.legend-pills .lp.full { background:#4b5563; }
.legend-pills .lp.partial { background:#4b5563; opacity:0.4; }
.legend-pills .lp.empty { background:transparent; border:1.5px solid #d1d5db; }
"""

# --- HTML ---
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HNIVision · Five methods on one image</title>
<style>{CSS}</style>
</head>
<body>
<div class="gallery">

<header class="gallery-header">
  <h1>Five computer vision methods on one image</h1>
  <p>Each column shows what one method 'saw' and what it produced — same input, different outputs</p>
</header>

<div class="cards-row">

  <!-- Card 1: Labeling -->
  <article class="method-card" style="--card-color:{COLORS[1]}">
    <div class="card-header"><span class="card-index">01</span><h2 class="card-title">Image labeling</h2></div>
    <div class="card-body">
      <div class="card-backend">Google Vision API · WordNet filter</div>
      <img class="card-image" src="01_original.jpg" alt="">
      <div class="card-image-caption">Input image (unchanged)</div>
      <div class="output-label">OUTPUT</div>
      {card1_bars}
    </div>
    <div class="card-footer">
      <div class="hni-levels-label">HNI LEVELS CAPTURED</div>
      {hni_levels_html(COVERAGE[1])}
    </div>
  </article>

  <!-- Card 2: Classification -->
  <article class="method-card" style="--card-color:{COLORS[2]}">
    <div class="card-header"><span class="card-index">02</span><h2 class="card-title">Image classification</h2></div>
    <div class="card-body">
      <div class="card-backend">ResNet-50 fine-tuned · 7 scenario classes</div>
      <img class="card-image" src="01_original.jpg" alt="">
      <div class="card-image-caption">Input image (unchanged)</div>
      <div class="output-label">OUTPUT</div>
      <div class="top-class-card">
        <div class="top-class-name">{top_class}</div>
        <div class="top-class-prob">p = {top_prob:.3f}</div>
      </div>
      <div class="other-classes-h">Other classes</div>
      {card2_others}
    </div>
    <div class="card-footer">
      <div class="hni-levels-label">HNI LEVELS CAPTURED</div>
      {hni_levels_html(COVERAGE[2])}
    </div>
  </article>

  <!-- Card 3: Detection -->
  <article class="method-card" style="--card-color:{COLORS[3]}">
    <div class="card-header"><span class="card-index">03</span><h2 class="card-title">Object detection</h2></div>
    <div class="card-body">
      <div class="card-backend">YOLO11s · Ultralytics (COCO 80)</div>
      <img class="card-image" src="detection_overlay.png" alt="">
      <div class="card-image-caption">Output: bounding boxes drawn on input</div>
      <div class="output-label">OUTPUT</div>
      {card3_boxes}
      <div class="det-stats">
        <div class="det-stat-row"><span>Total person count</span><span class="det-stat-val">{total_dets}</span></div>
        <div class="det-stat-row"><span>Other classes</span><span class="det-stat-val">0</span></div>
        <div class="det-stat-row"><span>Avg confidence</span><span class="det-stat-val">{avg_conf:.2f}</span></div>
      </div>
    </div>
    <div class="card-footer">
      <div class="hni-levels-label">HNI LEVELS CAPTURED</div>
      {hni_levels_html(COVERAGE[3])}
    </div>
  </article>

  <!-- Card 4: Segmentation -->
  <article class="method-card" style="--card-color:{COLORS[4]}">
    <div class="card-header"><span class="card-index">04</span><h2 class="card-title">Image segmentation</h2></div>
    <div class="card-body">
      <div class="card-backend">SegFormer-b0 · ADE20K · {n_classes} classes</div>
      <img class="card-image" src="segmentation_overlay.png" alt="">
      <div class="card-image-caption">Output: per-pixel semantic overlay</div>
      <div class="output-label">OUTPUT</div>
      {card4_bars}
    </div>
    <div class="card-footer">
      <div class="hni-levels-label">HNI LEVELS CAPTURED</div>
      {hni_levels_html(COVERAGE[4])}
    </div>
  </article>

  <!-- Card 5: VLM -->
  <article class="method-card" style="--card-color:{COLORS[5]}">
    <div class="card-header"><span class="card-index">05</span><h2 class="card-title">Vision-LLM</h2></div>
    <div class="card-body">
      <div class="card-backend">Qwen3-Omni-Flash · structured prompt</div>
      <img class="card-image" src="01_original.jpg" alt="">
      <div class="card-image-caption">Input image + structured HNI prompt</div>
      <div class="output-label">SCENE SUMMARY</div>
      <div class="scene-summary">{vlm_summary}</div>
      <div class="hni-tag h"><div class="hni-tag-label">human</div><div class="hni-tag-value">{hum_display}</div></div>
      <div class="hni-tag n"><div class="hni-tag-label">nature</div><div class="hni-tag-value">{nat_display}</div></div>
      <div class="hni-tag a"><div class="hni-tag-label">activity</div><div class="hni-tag-value">{act_display}</div></div>
      <div class="hni-tag m"><div class="hni-tag-label">meaning</div><div class="hni-tag-value">{mng_display}</div></div>
      <div class="vlm-meta">strength: {hni_strength} · conf: {confidence} · {tokens} tokens</div>
    </div>
    <div class="card-footer">
      <div class="hni-levels-label">HNI LEVELS CAPTURED</div>
      {hni_levels_html(COVERAGE[5])}
    </div>
  </article>

</div>

<footer class="gallery-footer">
  <div><strong>HNI levels —</strong> H human · N nature · A activity · M meaning</div>
  <div class="legend-pills">
    <span><span class="lp full"></span>full</span>
    <span><span class="lp partial"></span>partial</span>
    <span><span class="lp empty"></span>none</span>
  </div>
</footer>

</div>
</body>
</html>
"""

out_path = GALLERY / "01_methods.html"
out_path.write_text(HTML)
print(f"✓ Saved {out_path}")
print(f"  Size: {out_path.stat().st_size:,} bytes")
print(f"  Card 1 labels shown: {len(labels_top)}")
print(f"  Card 2 top class: {top_class} ({top_prob:.3f})")
print(f"  Card 3 detections: {total_dets}")
print(f"  Card 4 top seg class: {seg_sorted[0][0]} ({seg_sorted[0][1]*100:.1f}%)")
print(f"  Card 5 VLM summary length: {len(vlm_summary)} chars")
