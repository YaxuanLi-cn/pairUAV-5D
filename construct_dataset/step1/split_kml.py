#!/usr/bin/env python3
"""Split each tour KML into one KML file per gx:FlyTo frame."""

from __future__ import annotations

import argparse
import copy
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
ATOM_NS = "http://www.w3.org/2005/Atom"
ET.register_namespace("", KML_NS)
ET.register_namespace("gx", GX_NS)
ET.register_namespace("kml", KML_NS)
ET.register_namespace("atom", ATOM_NS)


def _tour_id(path: Path, tour: ET.Element) -> str:
    """Return the stable output directory name for a source tour."""
    name = tour.findtext(f"{{{KML_NS}}}name") or path.stem
    match = re.search(r"(\d+)$", name.strip())
    return match.group(1).zfill(4) if match else name.strip()


def _find_tour(root: ET.Element, path: Path) -> ET.Element:
    tours = root.findall(f"{{{GX_NS}}}Tour")
    if len(tours) != 1:
        raise ValueError(f"{path}: expected exactly one gx:Tour, found {len(tours)}")
    return tours[0]


def split_tour(source: Path, output_root: Path, *, overwrite: bool = True) -> int:
    """Split one source KML and return the number of generated frames."""
    root = ET.parse(source).getroot()
    tour = _find_tour(root, source)
    playlist = tour.find(f"{{{GX_NS}}}Playlist")
    if playlist is None:
        raise ValueError(f"{source}: gx:Tour has no gx:Playlist")

    frames = list(playlist.findall(f"{{{GX_NS}}}FlyTo"))
    if not frames:
        raise ValueError(f"{source}: gx:Playlist has no gx:FlyTo frames")

    destination = output_root / _tour_id(source, tour)
    if overwrite and destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)

    # Build a small, valid KML document for every frame. Copying the original
    # root and tour metadata keeps the output usable by KML readers.
    for frame_id, frame in enumerate(frames):
        frame_root = ET.Element(root.tag, root.attrib)
        frame_tour = ET.SubElement(frame_root, tour.tag, tour.attrib)
        for child in tour:
            if child.tag == f"{{{GX_NS}}}Playlist":
                continue
            frame_tour.append(copy.deepcopy(child))
        frame_playlist = ET.SubElement(frame_tour, playlist.tag, playlist.attrib)
        frame_playlist.append(copy.deepcopy(frame))

        target = destination / f"{frame_id:06d}.kml"
        with target.open("wb") as handle:
            ET.ElementTree(frame_root).write(
                handle,
                encoding="utf-8",
                xml_declaration=True,
                short_empty_elements=True,
            )

    return len(frames)


def split_all(input_dir: Path, output_dir: Path, *, overwrite: bool = True) -> tuple[int, int]:
    sources = sorted(input_dir.glob("*.kml"))
    if not sources:
        raise FileNotFoundError(f"No .kml files found in {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    frames = 0
    failures: list[tuple[Path, Exception]] = []
    for source in sources:
        try:
            frames += split_tour(source, output_dir, overwrite=overwrite)
            processed += 1
        except (ET.ParseError, OSError, ValueError) as exc:
            failures.append((source, exc))

    if failures:
        details = "\n".join(f"  {path}: {error}" for path, error in failures[:20])
        suffix = "" if len(failures) <= 20 else f"\n  ... and {len(failures) - 20} more"
        raise RuntimeError(
            f"Failed to process {len(failures)} of {len(sources)} files:\n{details}{suffix}"
        )
    return processed, frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/root/origin_tour"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/split_kml"))
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Keep existing tour directories instead of replacing them.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        processed, frames = split_all(
            args.input_dir,
            args.output_dir,
            overwrite=not args.no_overwrite,
        )
    except (FileNotFoundError, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        return 1

    print(f"Processed {processed} tours and wrote {frames} frames to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
