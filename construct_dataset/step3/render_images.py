#!/usr/bin/env python3
"""Render generated pair KML files through the step2 renderer."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

STEP2_DIR = Path(__file__).resolve().parents[1] / "step2"
sys.path.insert(0, str(STEP2_DIR))
from render_satellite import render  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs-dir", type=Path, default=Path("/root/processed_paired_json/0000"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/processed_images/0000"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/root/.cache/construct_dataset/google_tiles"))
    parser.add_argument("--min-tile-zoom", type=int, default=20, help="Never lower tile detail below this level")
    parser.add_argument("--image-size", type=int, default=1024, help="Square output resolution; camera field of view stays fixed")
    parser.add_argument("--supersample", type=int, default=2, choices=(1,2,3,4))
    parser.add_argument("--jpeg-quality", type=int, default=98)
    args = parser.parse_args()
    if args.image_size <= 0:
        parser.error("image-size must be positive")
    kml_dir = args.pairs_dir / "kml"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="step3-kml-", dir=args.output_dir.parent) as temp:
        input_dir = Path(temp) / "frames"
        input_dir.mkdir()
        for source in sorted(kml_dir.glob("*.kml")):
            target = input_dir / source.name
            target.symlink_to(source.resolve())
        image_dir = Path(temp) / "images"
        rendered = render(input_dir, image_dir, args.cache_dir, (args.image_size, args.image_size), per_frame_mosaic=True,
                          min_tile_zoom=args.min_tile_zoom, supersample=args.supersample, jpeg_quality=args.jpeg_quality)
        expected_names = {f"{path.stem}.jpeg" for path in kml_dir.glob("candidate_*.kml")}
        actual_names = {path.name for path in image_dir.glob("*.jpeg")}
        if actual_names != expected_names:
            raise RuntimeError("Rendered image set does not match candidate KMLs")
        for path in image_dir.glob("*.jpeg"):
            path.replace(args.output_dir / path.name)
        for old in args.output_dir.glob("candidate_*.jpeg"):
            if old.name not in expected_names:
                old.unlink()

    pairs_path = args.pairs_dir / "pairs.jsonl"
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines()]
    missing = []
    candidate_files = sorted(kml_dir.glob("candidate_*.kml"))
    for candidate in candidate_files:
        expected = args.output_dir / f"{candidate.stem}.jpeg"
        if not expected.exists():
            missing.append(str(expected))
    if missing:
        raise RuntimeError(f"missing {len(missing)} rendered pair images, sample={missing[:3]}")
    with pairs_path.open("w", encoding="utf-8") as handle:
        for pair in pairs:
            pair["source_image"] = Path(pair["source_kml"]).with_suffix(".jpeg").name
            pair["target_image"] = Path(pair["target_kml"]).with_suffix(".jpeg").name
            handle.write(json.dumps(pair, ensure_ascii=False, sort_keys=True) + "\n")
    manifest_path = args.pairs_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest["image_count"] = rendered
    manifest["image_size"] = [args.image_size, args.image_size]
    manifest["landmark_footprint_reference_size"] = [512, 512]
    manifest["render_quality"] = {"supersample": args.supersample, "jpeg_quality": args.jpeg_quality,
                                  "jpeg_subsampling": "4:4:4", "tile_cache": "original server bytes",
                                  "downsampling": "Lanczos"}
    manifest["minimum_tile_zoom"] = args.min_tile_zoom
    manifest["image_directory"] = str(args.output_dir.resolve())
    manifest["renderer"] = "satellite ground-plane projection (not a 3D terrain renderer)"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {rendered} generated KML files to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
