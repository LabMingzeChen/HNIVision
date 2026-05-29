"""Pre-compute 5-method pipeline outputs for HF Space gallery examples.

Reads images from space/examples/, runs the pipeline on each, and saves
the merged + per-method outputs as <stem>.json next to each image.

Skips images that already have a cached JSON.
"""
import json
import time
from pathlib import Path

from hnivision import Pipeline

EXAMPLES = Path("space/examples")
IMG_EXTS = {".jpg", ".jpeg", ".png"}

images = sorted([p for p in EXAMPLES.iterdir() if p.suffix.lower() in IMG_EXTS])
todo = [p for p in images if not p.with_suffix(".json").exists()]

if not todo:
    print(f"✓ All {len(images)} examples already cached, nothing to do.")
    raise SystemExit(0)

print(f"Found {len(images)} images, {len(todo)} need computing.")
print("Loading pipeline (warms all 5 methods)...")
pipe = Pipeline()
print()

for i, img in enumerate(todo, 1):
    print(f"[{i}/{len(todo)}] {img.name} ...", flush=True)
    t = time.time()
    out = pipe.extract(str(img))
    elapsed = time.time() - t

    data = {
        "merged": out.merged.model_dump(),
        "per_method": {n: r.model_dump() for n, r in out.per_method.items()},
    }
    json_path = img.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2, default=str)

    print(f"  ✓ {elapsed:.1f}s → {json_path.name}")

print(f"\n✓ Done. Cached {len(todo)} examples.")
