#!/usr/bin/env bash
# Reproduce tour 0000's accepted close-up dataset in a separate output tree.
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$project_root"
python_bin="${PYTHON_BIN:-/venv/main/bin/python}"
output_root="${1:-$project_root/reproduced_step3}"
cache_dir="${TILE_CACHE_DIR:-$project_root/.cache/construct_dataset/google_tiles}"
mkdir -p "$output_root"
output_root="$(cd -- "$output_root" && pwd)"
pairs_dir="$output_root/processed_paired_json/0000"
images_dir="$output_root/processed_images/0000"

"$python_bin" -m construct_dataset.step3.generate_pairs \
  --tour-id 0000 --first-key "$project_root/first-key/0000.kml" \
  --seed 42 --seed-version step3-v4-closeup \
  --candidate-count 70 --max-candidate-count 100 --solver-seconds 45 \
  --landmark-radius 60 --camera-height 180 --max-camera-height 210 \
  --horizontal-fov 70 --max-tilt 25 --min-landmark-pixels 200 \
  --output-dir "$pairs_dir"

"$python_bin" -m construct_dataset.step3.render_images \
  --pairs-dir "$pairs_dir" --output-dir "$images_dir" --cache-dir "$cache_dir" \
  --min-tile-zoom 20 --image-size 1024 --supersample 2 --jpeg-quality 98

"$python_bin" -m construct_dataset.step3.visualize_pairs \
  --pairs "$pairs_dir/pairs.jsonl" --output-dir "$pairs_dir/visualizations"

"$python_bin" -m construct_dataset.step3.validate_pairs \
  --pairs-dir "$pairs_dir" --image-dir "$images_dir"

"$python_bin" -m unittest construct_dataset.step3.test_geometry -v

"$python_bin" - "$output_root" <<'PY'
import json, sys
from importlib.metadata import version
from pathlib import Path
root = Path(sys.argv[1])
info = {'python': sys.version, 'packages': {p: version(p) for p in ('numpy', 'scipy', 'Pillow', 'matplotlib')}}
(root / 'environment.json').write_text(json.dumps(info, indent=2) + '\n')
print(f'Reproduction completed: {root}')
PY
