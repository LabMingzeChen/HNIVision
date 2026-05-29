"""HNIVision Gradio demo for Hugging Face Spaces.

Smart caching: gallery examples have pre-computed outputs (instant load).
Uploaded images run live inference (~30s on CPU).
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import gradio as gr

# ============================================================
# Secrets bootstrapping
# ============================================================
# Google Vision credentials: stored as base64-encoded JSON in HF secret
# GOOGLE_APPLICATION_CREDENTIALS_B64. Decode and write to disk at boot.
_GCP_B64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_B64")
if _GCP_B64:
    _GCP_PATH = Path("/tmp/gcp_creds.json")
    _GCP_PATH.write_bytes(base64.b64decode(_GCP_B64))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_GCP_PATH)

EXAMPLES_DIR = Path(__file__).parent / "examples"

# ============================================================
# Lazy pipeline (don't pay cold-start at app boot)
# ============================================================
_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from hnivision import Pipeline
        _pipeline = Pipeline()
    return _pipeline


# ============================================================
# Smart extract: cache lookup, else live inference
# ============================================================
def smart_extract(image_path):
    if image_path is None:
        return ("⚠️ Please select a gallery example or upload an image.",
                "", {}, {}, {}, {}, {})

    stem = Path(image_path).stem
    cache = EXAMPLES_DIR / f"{stem}.json"

    if cache.exists():
        with open(cache) as f:
            data = json.load(f)
        status = "⚡ **Loaded from pre-computed cache** (gallery example, instant)"
    else:
        start = time.time()
        pipe = get_pipeline()
        out = pipe.extract(image_path)
        elapsed = time.time() - start
        data = {
            "merged": out.merged.model_dump(),
            "per_method": {n: r.model_dump() for n, r in out.per_method.items()},
        }
        status = f"✓ **Live inference completed in {elapsed:.1f}s**"

    merged = data["merged"]
    per = data["per_method"]
    return (
        status,
        format_hni_summary(merged),
        per.get("labeling", {}),
        per.get("classification", {}),
        per.get("detection", {}),
        per.get("segmentation", {}),
        per.get("vlm", {}),
    )


def format_hni_summary(m):
    summary = m.get("summary") or "*No summary available*"
    human = m.get("human") or {}
    nature = m.get("nature") or {}
    activity = m.get("activity") or {}
    meaning = m.get("meaning") or {}

    def _tags(d):
        tags = d.get("tags") or []
        return ", ".join(tags) if tags else "—"

    return f"""### 📋 HNI Summary

> {summary}

| Layer | Tags | Dominant |
|---|---|---|
| 👤 **Human** | {_tags(human)} | — |
| 🌿 **Nature** | {_tags(nature)} | **{nature.get('dominant') or '—'}** |
| 🏃 **Activity** | {_tags(activity)} | **{activity.get('dominant') or '—'}** |
| 💭 **Meaning** | {_tags(meaning)} | **{meaning.get('dominant') or '—'}** |

**HNI present:** {m.get('hni_present', '?')} · **Strength:** {m.get('hni_strength', '?')} · **Source:** `{m.get('source_method', '?')}`
"""


# ============================================================
# UI
# ============================================================
example_images = sorted(
    [p for p in EXAMPLES_DIR.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
)

with gr.Blocks(title="HNIVision Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🌳 HNIVision — Five vision methods for Human-Nature Interaction

A unified pipeline that runs **5 computer vision methods** on the same image and merges their outputs into a **4-layer HNI schema** (`human` · `nature` · `activity` · `meaning`).

- **Click a gallery example** → instant pre-computed results
- **Upload your own image** → live inference (~30s on CPU free tier)

🔗 [GitHub repo](https://github.com/LabMingzeChen/HNIVision) · [Paper draft (TBD)]() · [ResNet weights](https://huggingface.co/Mingze/HNIVision-ResNet50)
""")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Image", height=320)
            run_btn = gr.Button("Run 5-method pipeline", variant="primary", size="lg")
            status_box = gr.Markdown()
            gr.Examples(
                examples=[[str(p)] for p in example_images],
                inputs=image_input,
                label="Gallery examples (click for instant pre-computed results)",
                examples_per_page=10,
            )

        with gr.Column(scale=2):
            summary_md = gr.Markdown()
            with gr.Accordion("Per-method raw outputs", open=False):
                with gr.Tabs():
                    with gr.Tab("🏷️ Labeling"):
                        labeling_json = gr.JSON(show_label=False)
                    with gr.Tab("📊 Classification"):
                        classification_json = gr.JSON(show_label=False)
                    with gr.Tab("📦 Detection"):
                        detection_json = gr.JSON(show_label=False)
                    with gr.Tab("🎨 Segmentation"):
                        segmentation_json = gr.JSON(show_label=False)
                    with gr.Tab("🤖 Vision-LLM"):
                        vlm_json = gr.JSON(show_label=False)

    run_btn.click(
        smart_extract,
        inputs=image_input,
        outputs=[status_box, summary_md, labeling_json, classification_json,
                 detection_json, segmentation_json, vlm_json],
    )

    gr.Markdown("""
---
**Methods used:**
🏷️ Google Vision API · 📊 ResNet-50 (7-class) · 📦 YOLO26-s · 🎨 SAM2 + SegFormer (ADE20K) · 🤖 GPT-4.1-mini
""")

if __name__ == "__main__":
    demo.launch()
