#!/usr/bin/env python3
"""Render assets/gallery/01_methods.html → 01_methods.png via headless Chromium."""

from pathlib import Path
from playwright.sync_api import sync_playwright

REPO = Path(__file__).parent.parent.resolve()
HTML_PATH = REPO / "assets" / "gallery" / "01_methods.html"
OUT_PATH = REPO / "assets" / "gallery" / "01_methods.png"

if not HTML_PATH.exists():
    raise SystemExit(
        f"❌ {HTML_PATH} not found.\n"
        f"   Run: python scripts/generate_gallery_html.py"
    )

print(f"Rendering {HTML_PATH.name} → {OUT_PATH.name}...")

with sync_playwright() as p:
    browser = p.chromium.launch()
    context = browser.new_context(
        viewport={"width": 1700, "height": 1500},
        device_scale_factor=2,
    )
    page = context.new_page()
    page.goto(f"file://{HTML_PATH}")
    page.wait_for_load_state("networkidle")
    
    gallery = page.locator(".gallery")
    gallery.screenshot(path=str(OUT_PATH))
    
    browser.close()

size = OUT_PATH.stat().st_size
print(f"✓ Saved {OUT_PATH}")
print(f"  File size: {size:,} bytes ({size/1024:.1f} KB)")

from PIL import Image
with Image.open(OUT_PATH) as im:
    print(f"  Dimensions: {im.size[0]} × {im.size[1]} pixels (2x retina)")
