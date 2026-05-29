"""Pre-compute 5-method outputs + overlay PNGs for HF Space gallery."""
import json
import time
from pathlib import Path

from hnivision import Pipeline
from hnivision.overlays import render_detection_overlay, render_segmentation_overlay

EXAMPLES = Path("space/examples")
IMG_EXTS = {".jpg", ".jpeg", ".png"}

images = sorted([
    p for p in EXAMPLES.iterdir()
    if p.suffix.lower() in IMG_EXTS
    and not p.stem.endswith(("_detection", "_segmentation"))
])

def needs_work(img):
    return not all([
        img.with_suffix(".json").exists(),
        (img.parent / f"{img.stem}_detection.png").exists(),
        (img.parent / f"{img.stem}_segmentation.png").exists(),
    ])

todo = [p for p in images if needs_work(p)]
if not todo:
    print(f"All {len(images)} examples have JSON + 2 overlays cached.")
    raise SystemExit(0)

print(f"Found {len(images)} images; {len(todo)} need processing.")
print("Loading pipeline (5 methods)...")
pipe = Pipeline()
print()

for i, img in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {img.name} ...", flush=True)
    t = time.time()
    out = pipe.extract(str(img))
    elapsed = time.time() - t

    data = {
        "merged": out.merged.model_dump(mode="json"),
        "per_method": {n: r.model_dump(mode="json") for n, r in out.per_method.items()},
    }
    json_path = img.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    boxes = (data["per_method"]["detection"].get("raw_output") or {}).get("boxes", [])
    det_path = img.parent / f"{img.stem}_detection.png"
    render_detection_overlay(img, boxes, det_path)

    raw_masks = getattr(pipe._methods["segmentation"], "_last_raw_masks", None)
    seg_path = img.parent / f"{img.stem}_segmentation.png"
    if raw_masks:
        render_segmentation_overlay(img, raw_masks, seg_path)
    else:
        print(f"  WARN: no _last_raw_masks on Segmentation, skipping {seg_path.name}")

    print(f"  done {elapsed:.1f}s -> {json_path.name}, {det_path.name}, {seg_path.name}")

print(f"\nDone. Processed {len(todo)} examples.")
