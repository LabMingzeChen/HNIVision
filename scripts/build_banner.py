#!/usr/bin/env python3
"""Build hero-banner.html + render hero-banner.png matching Phase 1 design.

Closely mimics the original banner layout. Uses the actual logo SVG
from assets/logo/, real method data, and GPT-4 as VLM backend label.
"""

from pathlib import Path
from playwright.sync_api import sync_playwright

REPO = Path(__file__).parent.parent.resolve()
BANNER_DIR = REPO / "assets" / "banner"
HTML_PATH = BANNER_DIR / "hero-banner.html"
PNG_PATH = BANNER_DIR / "hero-banner.png"

# ============================================================
# CSS — mimics Phase 1 banner aesthetic
# ============================================================
CSS = """
:root {
  --c1:#2D7D4F; --c1bg:rgba(45,125,79,0.10);
  --c2:#4A6B3A; --c2bg:rgba(74,107,58,0.10);
  --c3:#2563EB; --c3bg:rgba(37,99,235,0.10);
  --c4:#B45309; --c4bg:rgba(180,83,9,0.10);
  --c5:#B91C1C; --c5bg:rgba(185,28,28,0.10);
  --bg-cream:#f3efe5; --brand-dark:#14532d;
  --t1:#1f2937; --t2:#6b7280; --t3:#9ca3af; --border:#e5e7eb;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  -webkit-font-smoothing: antialiased; }

.banner {
  width: 1200px; height: 400px; position: relative;
  background: linear-gradient(135deg, #f5f2e8 0%, #ede8da 100%);
  display: flex;
}

/* Decorative bottom strip — green solid line + dashed orange */
.banner::after {
  content: ""; position: absolute; bottom: 0; left: 0; right: 0; height: 6px;
  background:
    linear-gradient(to bottom, transparent 0 2px, var(--brand-dark) 2px 4px, transparent 4px 6px),
    repeating-linear-gradient(90deg, #C68A3A 0 5px, transparent 5px 11px);
  background-size: 100% 100%, 100% 6px;
  background-position: 0 0, 0 6px;
  background-repeat: no-repeat, repeat-x;
}

/* === LEFT SIDE === */
.left { flex: 0 0 540px; padding: 36px 28px 40px 44px;
  display: flex; flex-direction: column; }

.brand { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
.brand-logo { width: 56px; height: 56px; display: block; }
.brand-name { font-size: 36px; font-weight: 800; letter-spacing: -1.2px; line-height: 1; }
.brand-name .hni { color: var(--brand-dark); }
.brand-name .vision { color: var(--t1); }

.tagline { font-size: 17px; color: var(--t1); line-height: 1.4;
  margin-bottom: 12px; font-weight: 500; }
.description { font-size: 12px; color: var(--t2); line-height: 1.55;
  margin-bottom: 18px; max-width: 410px; }

.stat-tiles { display: flex; gap: 8px; margin-top: auto; }
.stat-tile {
  background: #fff; border: 1px solid var(--border); border-radius: 7px;
  padding: 8px 11px; min-width: 66px; text-align: center;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03);
}
.stat-tile .n { font-size: 17px; font-weight: 800; color: var(--brand-dark); line-height: 1.1; }
.stat-tile .l { font-size: 8px; color: var(--t3); letter-spacing: 0.4px;
  text-transform: uppercase; margin-top: 2px; }

/* === RIGHT SIDE === */
.right { flex: 1; padding: 32px 36px 30px 16px;
  display: flex; flex-direction: column; }

.r-header { margin-bottom: 10px; }
.r-title { font-size: 13px; font-weight: 700; color: var(--t1); }
.r-subtitle { font-size: 9.5px; color: var(--t3); margin-top: 2px; }

.cards { display: flex; gap: 5px; flex: 1; }

/* Mini cards */
.mc { flex: 1; background: #fff; border: 1px solid var(--border); border-radius: 6px;
  overflow: hidden; display: flex; flex-direction: column;
  box-shadow: 0 1px 2px rgba(0,0,0,0.03); }
.mc-header { background: var(--mc-color); color: #fff; padding: 6px 8px 7px; position: relative; }
.mc-index { position: absolute; top: 3px; right: 6px; font-size: 7px; opacity: 0.7; font-weight: 500; }
.mc-title { font-size: 10.5px; font-weight: 700; }

.mc-body { padding: 5px 6px 6px; display: flex; flex-direction: column; flex: 1; }
.mc-backend { font-size: 7.5px; color: var(--mc-color); margin-bottom: 4px; font-weight: 600; }
.mc-img { width: 100%; aspect-ratio: 800 / 532; height: auto; object-fit: cover; border-radius: 2px; margin-bottom: 4px; }
.mc-out-label { font-size: 6px; font-weight: 700; color: var(--t3); letter-spacing: 0.4px; margin-bottom: 1px; }
.mc-out-title { font-size: 8.5px; font-weight: 600; color: var(--t1); margin-bottom: 4px; line-height: 1.15; }
.mc-data { background: var(--mc-bg); padding: 5px 6px; border-radius: 3px;
  font-size: 7.5px; color: var(--mc-color); line-height: 1.5; flex: 1;
  font-family: "SF Mono", ui-monospace, monospace; }
.mc-cov { margin-top: 4px; padding-top: 3px; }
.mc-cov-label { font-size: 5.5px; font-weight: 700; color: var(--t3); letter-spacing: 0.4px;
  margin-bottom: 2px; text-transform: uppercase; }
.mc-dots { display: flex; gap: 3px; }
.mc-dot { width: 6.5px; height: 6.5px; border-radius: 50%; }
.mc-dot.full { background: var(--mc-color); }
.mc-dot.partial { background: var(--mc-color); opacity: 0.38; }
.mc-dot.empty { background: #fff; border: 1px solid #d1d5db; }

/* Bottom legend below cards */
.legend { margin-top: 7px; font-size: 7.5px; color: var(--t2);
  display: flex; gap: 9px; align-items: center; }
.legend .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--t2);
  display: inline-block; margin-right: 3px; vertical-align: middle; }
"""

# ============================================================
# Card data (REAL outputs + GPT-4 as VLM backend)
# ============================================================
CARDS = [
    {
        "index": "01", "title": "Labeling", "color": "var(--c1)", "bg": "var(--c1bg)",
        "backend": "Google Vision",
        "img": "../gallery/01_original.jpg",
        "out_title": "Multi-label tags",
        "data": "Cloud · 0.92<br>Leisure · 0.85<br>People in nature · 0.83<br>Walkway · 0.71",
        "coverage": ["partial", "partial", "partial", "partial"],
    },
    {
        "index": "02", "title": "Classification", "color": "var(--c2)", "bg": "var(--c2bg)",
        "backend": "ResNet-50 · 7 classes",
        "img": "../gallery/01_original.jpg",
        "out_title": "Top-1 category",
        "data": "<b>Garden Vegetation</b><br>p = 0.60<br>(top-1 — scene is<br>actually waterfront)",
        "coverage": ["empty", "full", "empty", "partial"],
    },
    {
        "index": "03", "title": "Detection", "color": "var(--c3)", "bg": "var(--c3bg)",
        "backend": "YOLO26-s · COCO",
        "img": "../gallery/detection_overlay.png",
        "out_title": "Bounding boxes",
        "data": "person · 0.89<br>person · 0.79<br>count: 2<br>avg conf: 0.84",
        "coverage": ["full", "empty", "empty", "empty"],
    },
    {
        "index": "04", "title": "Segmentation", "color": "var(--c4)", "bg": "var(--c4bg)",
        "backend": "SAM2 + SegFormer · ADE20K",
        "img": "../gallery/segmentation_overlay.png",
        "out_title": "Per-pixel labels",
        "data": "sky 46% · tree 19%<br>path 14% · bannister 11%<br>grass 9% · person 2%<br>(60 masks · 8 cls)",
        "coverage": ["full", "full", "empty", "empty"],
    },
    {
        "index": "05", "title": "Vision-LLM", "color": "var(--c5)", "bg": "var(--c5bg)",
        "backend": "GPT-4 · JSON prompt",
        "img": "../gallery/01_original.jpg",
        "out_title": "Structured JSON",
        "data": "human: 2-5 individuals<br>nature: waterfront<br>activity: viewing<br>meaning: aesthetic",
        "coverage": ["full", "full", "full", "full"],
    },
]

def render_card(c):
    dots = "".join([f'<span class="mc-dot {s}"></span>' for s in c["coverage"]])
    return f"""
    <div class="mc" style="--mc-color:{c['color']}; --mc-bg:{c['bg']};">
      <div class="mc-header">
        <span class="mc-index">{c['index']}</span>
        <div class="mc-title">{c['title']}</div>
      </div>
      <div class="mc-body">
        <div class="mc-backend">{c['backend']}</div>
        <img class="mc-img" src="{c['img']}" alt="">
        <div class="mc-out-label">OUTPUT</div>
        <div class="mc-out-title">{c['out_title']}</div>
        <div class="mc-data">{c['data']}</div>
        <div class="mc-cov">
          <div class="mc-cov-label">HNI COVERAGE</div>
          <div class="mc-dots">{dots}</div>
        </div>
      </div>
    </div>"""

cards_html = "\n".join(render_card(c) for c in CARDS)

# ============================================================
# Logo: try mark-primary.svg, fall back to PNG
# ============================================================
logo_candidates = [
    REPO / "assets" / "logo" / "mark-primary.svg",
    REPO / "assets" / "logo" / "mark-primary.png",
    REPO / "assets" / "logo" / "mark-primary-dark.svg",
]
logo_path = next((p for p in logo_candidates if p.exists()), None)
if logo_path is None:
    raise SystemExit("❌ No logo found in assets/logo/")
logo_rel = f"../logo/{logo_path.name}"
print(f"Using logo: {logo_rel}")

# ============================================================
# Assemble HTML
# ============================================================
HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>HNI Vision banner</title>
<style>{CSS}</style>
</head>
<body>
<div class="banner">
  <div class="left">
    <div class="brand">
      <img class="brand-logo" src="{logo_rel}" alt="HNI Vision">
      <div class="brand-name"><span class="hni">HNI</span> <span class="vision">Vision</span></div>
    </div>
    <div class="tagline">Five computer vision methods.<br>One framework for human-nature interaction.</div>
    <div class="description">Open-source Python toolkit + benchmark for extracting human-nature interaction from social media imagery. Five vision models, one unified API.</div>
    <div class="stat-tiles">
      <div class="stat-tile"><div class="n">5</div><div class="l">methods</div></div>
      <div class="stat-tile"><div class="n">4</div><div class="l">HNI levels</div></div>
      <div class="stat-tile"><div class="n">1,000</div><div class="l">images target</div></div>
      <div class="stat-tile"><div class="n">MIT</div><div class="l">license</div></div>
    </div>
  </div>
  <div class="right">
    <div class="r-header">
      <div class="r-title">Same input image, five methods</div>
      <div class="r-subtitle">NYC Fort Tryon Park 160710 — visitors overlooking lawn, trees, waterfront</div>
    </div>
    <div class="cards">{cards_html}</div>
    <div class="legend">
      <span><span class="dot"></span>human</span>
      <span><span class="dot"></span>nature</span>
      <span><span class="dot"></span>activity</span>
      <span><span class="dot"></span>meaning</span>
    </div>
  </div>
</div>
</body>
</html>
"""

HTML_PATH.write_text(HTML)
print(f"✓ Wrote {HTML_PATH.name} ({HTML_PATH.stat().st_size:,} bytes)")

# ============================================================
# Render via Playwright
# ============================================================
print(f"Rendering with Playwright → {PNG_PATH.name}...")
with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context(
        viewport={"width": 1200, "height": 400},
        device_scale_factor=2,  # 2x retina → 2400x800
    )
    page = context.new_page()
    page.goto(f"file://{HTML_PATH}")
    page.wait_for_load_state("networkidle")
    page.locator(".banner").screenshot(path=str(PNG_PATH))
    browser.close()

from PIL import Image
with Image.open(PNG_PATH) as im:
    print(f"✓ Saved {PNG_PATH.name} ({PNG_PATH.stat().st_size:,} bytes)")
    print(f"  Dimensions: {im.size[0]} × {im.size[1]}")
