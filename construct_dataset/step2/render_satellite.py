#!/usr/bin/env python3
"""Render split KML LookAt frames as Google satellite JPEG images.

The renderer uses the same Google satellite tile service used by Earth and
projects each KML camera onto a clean, UI-free image. Tiles are cached locally
so frames from one tour only download imagery once.
"""

from __future__ import annotations

import argparse
import io
import math
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
TILE_SIZE = 256
EARTH_RADIUS_M = 6_378_137.0
EARTH_CIRCUMFERENCE_M = 2 * math.pi * EARTH_RADIUS_M
TILE_SERVERS = (0, 1, 2, 3)
USER_AGENT = "construct-dataset-step2/1.0"


def parse_frame(path: Path) -> dict[str, float]:
    root = ET.parse(path).getroot()
    look = root.find(f".//{{{KML_NS}}}LookAt")
    if look is None:
        raise ValueError(f"{path}: no LookAt element")

    def value(name: str, default: float | None = None) -> float:
        text = look.findtext(f"{{{KML_NS}}}{name}")
        if text is None:
            if default is None:
                raise ValueError(f"{path}: LookAt has no {name}")
            return default
        return float(text)

    return {
        "longitude": value("longitude"),
        "latitude": value("latitude"),
        "altitude": value("altitude", 0.0),
        "heading": value("heading", 0.0) % 360.0,
        "tilt": value("tilt", 0.0),
        "range": max(value("range", 256.0), 1.0),
        "horiz_fov": float(look.findtext(f"{{{GX_NS}}}horizFov") or 60.0),
    }


def world_pixel(longitude: float, latitude: float, zoom: int) -> tuple[float, float]:
    scale = TILE_SIZE * (2**zoom)
    x = (longitude + 180.0) / 360.0 * scale
    bounded_lat = max(-85.05112878, min(85.05112878, latitude))
    # Slippy-map/Web Mercator projection: the isometric latitude uses tan(phi).
    y = (0.5 - math.asinh(math.tan(math.radians(bounded_lat))) / (2 * math.pi)) * scale
    return x, y


def meters_per_pixel(latitude: float, zoom: int) -> float:
    return EARTH_CIRCUMFERENCE_M * math.cos(math.radians(latitude)) / (TILE_SIZE * (2**zoom))


def tile_url(x: int, y: int, zoom: int) -> str:
    n = 2**zoom
    return f"https://mt{(x + y) % len(TILE_SERVERS)}.google.com/vt/lyrs=s&x={x % n}&y={y}&z={zoom}"


def download_tile(x: int, y: int, zoom: int, cache_dir: Path) -> tuple[int, int, Image.Image]:
    # Keep the server's original bytes in a versioned cache. Legacy entries
    # were recompressed at JPEG quality 92 and cannot be recovered by resaving.
    target = cache_dir / "original" / f"{zoom}_{x % (2**zoom)}_{y}.jpg"
    if target.exists() and target.stat().st_size > 1000:
        return x, y, Image.open(target).convert("RGB")

    request = urllib.request.Request(tile_url(x, y, zoom), headers={"User-Agent": USER_AGENT})
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read()
            image = Image.open(io.BytesIO(data)).convert("RGB")
            target.parent.mkdir(parents=True, exist_ok=True)
            # Nearby tours can request the same tile concurrently. Publish a
            # complete response atomically so readers never open a partial JPEG.
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.tile-', delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(data)
            temporary.replace(target)
            return x, y, image
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
    raise RuntimeError(f"failed to download tile {x},{y},{zoom}: {last_error}")


def camera_ground_points(frame: dict[str, float], size: tuple[int, int]) -> list[tuple[float, float]]:
    """Return ground-plane east/north coordinates for the four image corners."""
    width, height = size
    horizontal = math.radians(frame["horiz_fov"])
    vertical = 2 * math.atan(math.tan(horizontal / 2) * height / width)
    tilt = math.radians(frame["tilt"])
    heading = math.radians(frame["heading"])
    distance = frame["range"] * math.sin(tilt)
    camera_height = frame["range"] * math.cos(tilt) + frame["altitude"]
    forward = (math.sin(heading), math.cos(heading))
    right = (math.cos(heading), -math.sin(heading))
    view = (forward[0] * distance / frame["range"], forward[1] * distance / frame["range"], -math.cos(tilt))
    up = (forward[0] * math.cos(tilt), forward[1] * math.cos(tilt), distance / frame["range"])
    camera = (-distance * forward[0], -distance * forward[1], camera_height)
    points = []
    for x in (-1.0, 1.0):
        for y in (-1.0, 1.0):
            ray = (
                view[0] + math.tan(horizontal / 2) * x * right[0] + math.tan(vertical / 2) * y * up[0],
                view[1] + math.tan(horizontal / 2) * x * right[1] + math.tan(vertical / 2) * y * up[1],
                view[2] + math.tan(vertical / 2) * y * up[2],
            )
            if ray[2] >= -1e-6:
                raise ValueError("camera ray does not intersect the ground plane")
            scale = -camera[2] / ray[2]
            points.append((camera[0] + scale * ray[0], camera[1] + scale * ray[1]))
    return points


def mosaic_bounds(frames: list[dict[str, float]], zoom: int, size: tuple[int, int]) -> tuple[int, int, int, int]:
    """Tile bounds of ALL shifted ground footprints, with no clipped rays."""
    xs, ys = [], []
    for frame in frames:
        center_x, center_y = world_pixel(frame["longitude"], frame["latitude"], zoom)
        mpp = meters_per_pixel(frame["latitude"], zoom)
        for east, north in camera_ground_points(frame, size):
            xs.append(center_x + east / mpp)
            ys.append(center_y - north / mpp)
    return (math.floor((min(xs)-2)/TILE_SIZE), math.floor((min(ys)-2)/TILE_SIZE),
            math.floor((max(xs)+2)/TILE_SIZE), math.floor((max(ys)+2)/TILE_SIZE))


def download_mosaic(frames: list[dict[str, float]], zoom: int, cache_dir: Path, size: tuple[int, int]) -> tuple[Image.Image, float, float]:
    left, top, right, bottom = mosaic_bounds(frames, zoom, size)
    xs, ys = range(left, right+1), range(top, bottom+1)
    mosaic = Image.new("RGB", (len(xs)*TILE_SIZE, len(ys)*TILE_SIZE))
    with ThreadPoolExecutor(max_workers=12) as pool:
        futures = [pool.submit(download_tile, x, y, zoom, cache_dir) for y in ys for x in xs]
        for future in as_completed(futures):
            x, y, image = future.result()
            mosaic.paste(image, ((x-left)*TILE_SIZE, (y-top)*TILE_SIZE))
    center_x, center_y = world_pixel(frames[0]["longitude"], frames[0]["latitude"], zoom)
    return mosaic, center_x-left*TILE_SIZE, center_y-top*TILE_SIZE


def render_frame(frame: dict[str, float], mosaic: Image.Image, origin_x: float, origin_y: float, zoom: int, size: tuple[int, int]) -> Image.Image:
    width, height = size
    mpp = meters_per_pixel(frame["latitude"], zoom)
    horizontal = math.radians(frame["horiz_fov"])
    vertical = 2 * math.atan(math.tan(horizontal / 2) * height / width)
    tilt = math.radians(frame["tilt"])
    heading = math.radians(frame["heading"])
    distance = frame["range"] * math.sin(tilt)
    camera_height = frame["range"] * math.cos(tilt) + frame["altitude"]
    forward = (math.sin(heading), math.cos(heading))
    right = (math.cos(heading), -math.sin(heading), 0.0)
    view = (forward[0] * distance / frame["range"], forward[1] * distance / frame["range"], -math.cos(tilt))
    up = (forward[0] * math.cos(tilt), forward[1] * math.cos(tilt), distance / frame["range"])
    camera = np.array([-distance * forward[0], -distance * forward[1], camera_height], dtype=np.float64)

    px = (np.arange(width, dtype=np.float64) + 0.5) / width * 2.0 - 1.0
    py = 1.0 - (np.arange(height, dtype=np.float64) + 0.5) / height * 2.0
    xx, yy = np.meshgrid(px, py)
    rays = np.empty((height, width, 3), dtype=np.float64)
    rays[:, :, 0] = view[0] + math.tan(horizontal / 2) * xx * right[0] + math.tan(vertical / 2) * yy * up[0]
    rays[:, :, 1] = view[1] + math.tan(horizontal / 2) * xx * right[1] + math.tan(vertical / 2) * yy * up[1]
    rays[:, :, 2] = view[2] + math.tan(vertical / 2) * yy * up[2]
    scale = -camera_height / rays[:, :, 2]
    east = camera[0] + scale * rays[:, :, 0]
    north = camera[1] + scale * rays[:, :, 1]
    sample_x = origin_x + east / mpp
    sample_y = origin_y - north / mpp

    source = np.asarray(mosaic)
    if (sample_x.min() < 0 or sample_y.min() < 0 or
            sample_x.max() >= source.shape[1]-1 or sample_y.max() >= source.shape[0]-1):
        raise ValueError("Frame extends beyond the tile mosaic; refusing to clamp/repeat image edges")
    sample_x = np.clip(sample_x, 0, source.shape[1] - 1.001)
    sample_y = np.clip(sample_y, 0, source.shape[0] - 1.001)
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, source.shape[1] - 1)
    y1 = np.minimum(y0 + 1, source.shape[0] - 1)
    wx = (sample_x - x0)[..., None]
    wy = (sample_y - y0)[..., None]
    top = source[y0, x0] * (1 - wx) + source[y0, x1] * wx
    bottom = source[y1, x0] * (1 - wx) + source[y1, x1] * wx
    return Image.fromarray(np.clip(top * (1 - wy) + bottom * wy, 0, 255).astype(np.uint8), mode="RGB")


def render(input_dir: Path, output_dir: Path, cache_dir: Path, size: tuple[int, int],
           per_frame_mosaic: bool = False, min_tile_zoom: int = 1,
           supersample: int = 1, jpeg_quality: int = 95) -> int:
    if supersample not in (1, 2, 3, 4):
        raise ValueError("supersample must be between 1 and 4")
    if not 1 <= jpeg_quality <= 100:
        raise ValueError("jpeg_quality must be between 1 and 100")
    if not 1 <= min_tile_zoom <= 20:
        raise ValueError("min_tile_zoom must be between 1 and 20")
    files = sorted(input_dir.glob("*.kml"))
    if not files:
        raise FileNotFoundError(f"no KML frames found in {input_dir}")
    frames = [parse_frame(path) for path in files]
    # Separate mosaics preserve local detail when unrelated headings have a
    # very large combined footprint. Tile files remain shared in the cache.
    groups = [([path], [frame]) for path, frame in zip(files, frames)] if per_frame_mosaic else [(files, frames)]
    pixel_budget = 100_000_000 if per_frame_mosaic else 36_000_000
    output_dir.mkdir(parents=True, exist_ok=True)
    completed = 0
    for group_files, group_frames in groups:
        zoom = 20
        while True:
            left, top, right, bottom = mosaic_bounds(group_frames, zoom, size)
            if (right-left+1)*(bottom-top+1)*TILE_SIZE**2 <= pixel_budget:
                break
            if zoom <= min_tile_zoom:
                raise ValueError("Footprint exceeds memory budget at minimum tile resolution; reduce the camera footprint instead of degrading landmark detail")
            zoom -= 1
        print(f"Using Google satellite tiles at zoom {zoom}; frames={len(group_files)}", flush=True)
        mosaic, origin_x, origin_y = download_mosaic(group_frames, zoom, cache_dir, size)
        anchor_x, anchor_y = world_pixel(group_frames[0]["longitude"], group_frames[0]["latitude"], zoom)
        for source, frame in zip(group_files, group_frames):
            center_x, center_y = world_pixel(frame["longitude"], frame["latitude"], zoom)
            render_size = (size[0]*supersample, size[1]*supersample)
            image = render_frame(frame, mosaic, origin_x+center_x-anchor_x, origin_y+center_y-anchor_y, zoom, render_size)
            if supersample > 1:
                image = image.resize(size, resample=Image.Resampling.LANCZOS)
            target = output_dir / f"{source.stem}.jpeg"
            image.save(target, format="JPEG", quality=jpeg_quality, subsampling=0, optimize=True)
            completed += 1
            print(f"{completed}/{len(files)} {target}", flush=True)
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/root/split_kml/0000"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/origin_images/0000"))
    parser.add_argument("--cache-dir", type=Path, default=Path("/root/.cache/construct_dataset/google_tiles"))
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args()
    try:
        count = render(args.input_dir, args.output_dir, args.cache_dir, (args.width, args.height))
    except (FileNotFoundError, ValueError, RuntimeError, urllib.error.URLError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Rendered {count} frames to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
