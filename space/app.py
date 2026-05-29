"""HNIVision Gradio demo for Hugging Face Spaces.

Method-grid UI: each of 5 methods has its own panel.
Detection + Segmentation cards include visual overlays.
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import gradio as gr

# --- Bootstrap GCP credentials from base64 secret ---
_GCP_B64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_B64")
if _GCP_B64:
    _GCP_PATH = Path("/tmp/gcp_creds.json")
    _GCP_PATH.write_bytes(base64.b64decode(_GCP_B64))
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(_GCP_PATH)

EXAMPLES_DIR = Path(__file__).parent / "examples"


def _f(x, default=0.0):
    """Safely cast to float."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


# --- Lazy pipeline ---
_pipeline = None
def get_pipeline():
    global _pipeline
    if _pipeline is None:
        from hnivision import Pipeline
        _pipeline = Pipeline()
    return _pipeline


def smart_extract(image_path):
    """Process image. Returns 9 outputs populating the UI."""
    if image_path is None:
        return ("⚠️ Please select a gallery example or upload an image.",
                "", "", "", "", None, "", None, "")

    stem = Path(image_path).stem
    cache = EXAMPLES_DIR / f"{stem}.json"
    det_cached = EXAMPLES_DIR / f"{stem}_detection.png"
    seg_cached = EXAMPLES_DIR / f"{stem}_segmentation.png"

    if cache.exists() and det_cached.exists() and seg_cached.exists():
        with open(cache) as f:
            data = json.load(f)
        status = "⚡ **Cached** (gallery example, instant)"
        det_path = str(det_cached)
        seg_path = str(seg_cached)
    else:
        start = time.time()
        pipe = get_pipeline()
        out = pipe.extract(image_path)
        elapsed = time.time() - start
        data = {
            "merged": out.merged.model_dump(mode="json"),
            "per_method": {n: r.model_dump(mode="json") for n, r in out.per_method.items()},
        }
        from hnivision.overlays import render_detection_overlay, render_segmentation_overlay
        tmp = Path("/tmp/hnivision_live"); tmp.mkdir(exist_ok=True)
        det_path = str(tmp / f"{stem}_detection.png")
        seg_path = str(tmp / f"{stem}_segmentation.png")
        det_raw = data["per_method"]["detection"].get("raw_output") or {}
        boxes = det_raw.get("boxes") or []
        render_detection_overlay(image_path, boxes, det_path)
        seg_method = pipe._methods.get("segmentation")
        raw_masks = getattr(seg_method, "_last_raw_masks", None) if seg_method else None
        if raw_masks:
            render_segmentation_overlay(image_path, raw_masks, seg_path)
        else:
            seg_path = None
        status = f"✓ **Live inference** completed in {elapsed:.1f}s"

    merged = data["merged"]
    per = data["per_method"]
    return (
        status,
        _fmt_summary(merged),
        _fmt_labeling(per.get("labeling", {})),
        _fmt_classification(per.get("classification", {})),
        _fmt_vlm(per.get("vlm", {})),
        det_path,
        _fmt_detection(per.get("detection", {})),
        seg_path,
        _fmt_segmentation(per.get("segmentation", {})),
    )


def _fmt_summary(m):
    def t(d): return ", ".join((d or {}).get("tags", [])) or "—"
    def dom(d): return (d or {}).get("dominant") or "—"
    src = m.get("source_method", "?")
    return f"""### 📋 HNI Summary

> {m.get("summary") or "*No summary*"}

| Layer | Tags | Dominant |
|---|---|---|
| 👤 **Human** | {t(m.get("human"))} | — |
| 🌿 **Nature** | {t(m.get("nature"))} | **{dom(m.get("nature"))}** |
| 🏃 **Activity** | {t(m.get("activity"))} | **{dom(m.get("activity"))}** |
| 💭 **Meaning** | {t(m.get("meaning"))} | **{dom(m.get("meaning"))}** |

**HNI present:** {m.get("hni_present", "?")} · **Strength:** {m.get("hni_strength", "?")}  
*Source: `{src[:80]}`*
"""


def _fmt_labeling(d):
    raw = d.get("raw_output") or {}
    all_labels = raw.get("labels") or []
    kept = [l for l in all_labels if l.get("kept_by_filter")]
    lines = ["### 🏷️ Labeling", "*Google Vision · WordNet filter*", "",
             f"**{len(kept)}/{len(all_labels)} labels** kept (WordNet match)", ""]
    if not kept:
        return "\n".join(lines + ["*No labels passed filter*"])
    for lbl in kept[:8]:
        name = lbl.get("description") or "?"
        conf = _f(lbl.get("confidence") or 0)
        lines.append(f"- `{name}` · {conf:.2f}")
    return "\n".join(lines)


def _fmt_classification(d):
    raw = d.get("raw_output") or {}
    pred = raw.get("predicted_class") or "?"
    conf = _f(raw.get("confidence"))
    lines = ["### 📊 Classification", "*ResNet-50 · 7 HNI scenarios*", "",
             f"**Top-1:** `{pred}` · p = {conf:.3f}", ""]
    probs = raw.get("all_probs") or {}
    if probs:
        sorted_probs = sorted(((c, _f(pp)) for c, pp in probs.items()),
                               key=lambda kv: kv[1], reverse=True)
        for cls, p in sorted_probs[:7]:
            bar = "▓" * max(1, int(p * 18))
            lines.append(f"`{cls[:24]:<24s}` {p:.2f} {bar}")
    return "\n".join(lines)


def _fmt_vlm(d):
    raw = d.get("raw_output") or {}
    lines = ["### 🤖 Vision-LLM", "*GPT-4.1-mini · structured JSON prompt*", "",
             f"> {(d.get('summary') or '*No summary*')[:180]}", ""]
    for layer in ["human", "nature", "activity", "meaning"]:
        ld = d.get(layer) or {}
        dom = ld.get("dominant")
        tags = ld.get("tags") or []
        if dom or tags:
            tag_str = ", ".join(tags[:3]) if tags else ""
            lines.append(f"- **{layer}:** `{dom or tag_str[:24]}`")
    tokens = raw.get("tokens_used") or d.get("tokens_used")
    if tokens:
        lines.append(f"\n*{tokens} tokens*")
    return "\n".join(lines)


def _fmt_detection(d):
    from collections import Counter
    raw = d.get("raw_output") or {}
    boxes = raw.get("boxes") or []
    lines = ["### 📦 Detection", "*YOLO26-s · Ultralytics (COCO 80)*", "",
             f"**{len(boxes)} object(s) detected**", ""]
    if not boxes:
        return "\n".join(lines + ["*No detections above threshold*"])
    # Per-class count + avg confidence
    counts = Counter(b.get("class_name", "?") for b in boxes)
    lines.append("**By class:**")
    for cls, cnt in counts.most_common():
        cls_confs = [_f(b.get("confidence", 0)) for b in boxes if b.get("class_name") == cls]
        avg_c = sum(cls_confs) / len(cls_confs)
        lines.append(f"- **{cnt}×** `{cls}` (avg conf {avg_c:.2f})")
    overall_avg = sum(_f(b.get("confidence", 0)) for b in boxes) / len(boxes)
    lines.append(f"\n*Overall avg confidence: {overall_avg:.2f}*")
    return "\n".join(lines)


def _fmt_segmentation(d):
    raw = d.get("raw_output") or {}
    shares = (raw.get("pixel_shares") or {}).get("shares") or {}
    n_masks = raw.get("n_masks", 0)
    n_classes = raw.get("n_classes_detected", 0)
    lines = ["### 🎨 Segmentation",
             "*SAM2 + SegFormer · ADE20K majority vote*", "",
             f"**{n_masks} instance masks · {n_classes} ADE20K classes**", ""]
    for cls, share_raw in list(shares.items())[:8]:
        share = _f(share_raw)
        bar = "▓" * max(1, int(share * 20))
        lines.append(f"`{cls[:18]:<18s}` {share*100:5.1f}% {bar}")
    return "\n".join(lines)


# --- UI ---
example_images = sorted([
    p for p in EXAMPLES_DIR.glob("*.jpg")
    if not p.stem.endswith(("_detection", "_segmentation"))
])

with gr.Blocks(title="HNIVision Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
# 🌳 HNIVision — Five vision methods for Human-Nature Interaction

A unified pipeline running **5 CV methods** on the same image → merged into a **4-layer HNI schema** (`human` · `nature` · `activity` · `meaning`).

- **Click a gallery example** → instant pre-computed results
- **Upload your own image** → live inference (~60-120s on CPU)

🔗 [GitHub](https://github.com/LabMingzeChen/HNIVision) · [ResNet weights](https://huggingface.co/Mingze/HNIVision-ResNet50)
""")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="filepath", label="Image", height=320)
            run_btn = gr.Button("Run 5-method pipeline", variant="primary", size="lg")
            status_box = gr.Markdown()
            gr.Examples(
                examples=[[str(p)] for p in example_images],
                inputs=image_input,
                label="Pre-computed gallery (click for instant)",
                examples_per_page=10,
            )
        with gr.Column(scale=2):
            summary_md = gr.Markdown()

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            labeling_md = gr.Markdown()
        with gr.Column():
            classification_md = gr.Markdown()
        with gr.Column():
            vlm_md = gr.Markdown()

    with gr.Row():
        with gr.Column():
            with gr.Group():
                detection_img = gr.Image(label="Detection overlay (YOLO26-s bboxes)",
                                          show_label=True, height=280)
                detection_md = gr.Markdown()
        with gr.Column():
            with gr.Group():
                segmentation_img = gr.Image(label="Segmentation overlay (SAM2 instance masks)",
                                             show_label=True, height=280)
                segmentation_md = gr.Markdown()

    gr.Markdown("""
---
**Backends:** 🏷️ Google Vision · 📊 ResNet-50 · 📦 YOLO26-s · 🎨 SAM2+SegFormer · 🤖 GPT-4.1-mini
""")

    run_btn.click(
        smart_extract,
        inputs=image_input,
        outputs=[
            status_box, summary_md,
            labeling_md, classification_md, vlm_md,
            detection_img, detection_md,
            segmentation_img, segmentation_md,
        ],
    )

if __name__ == "__main__":
    demo.launch()
