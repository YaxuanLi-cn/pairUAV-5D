#!/usr/bin/env python3
"""Capture 512x512 Google Earth Pro 3D frames for tour 0000."""

from __future__ import annotations

import math
import os
import shutil
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image

KML_NS = "http://www.opengis.net/kml/2.2"
GX_NS = "http://www.google.com/kml/ext/2.2"
SCREEN_X, SCREEN_Y = 341, 75
FRAME_SIZE = 512
DISPLAY_ENV = {"DISPLAY": ":150"}


def read_cameras(input_dir: Path) -> list[dict[str, float]]:
    cameras = []
    for path in sorted(input_dir.glob("*.kml"))[::5]:
        root = ET.parse(path).getroot()
        look = root.find(f".//{{{KML_NS}}}LookAt")
        cameras.append({
            "heading": float(look.findtext(f"{{{KML_NS}}}heading")) % 360,
            "range": float(look.findtext(f"{{{KML_NS}}}range")),
        })
    return cameras[:55]


def click(x: int, y: int, button: int = 1) -> None:
    subprocess.run(["xdotool", "mousemove", str(x), str(y), "click", str(button)], env={**os.environ, **DISPLAY_ENV}, check=True)


def key(name: str) -> None:
    subprocess.run(["xdotool", "key", name], env={**os.environ, **DISPLAY_ENV}, check=True)


def wheel(button: int, count: int) -> None:
    for _ in range(max(0, count)):
        click(600, 380, button)
        time.sleep(0.08)


def main() -> int:
    input_dir = Path("/root/split_kml/0000")
    output_dir = Path("/root/origin_images/0000")
    output_dir.mkdir(parents=True, exist_ok=True)
    cameras = read_cameras(input_dir)
    if len(cameras) != 55:
        raise RuntimeError(f"expected 55 sampled cameras, got {len(cameras)}")

    xvfb = subprocess.Popen(["Xvfb", ":150", "-screen", "0", "1280x720x24"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    earth = None
    try:
        time.sleep(1)
        env = os.environ.copy()
        env.update({"DISPLAY": ":150", "HOME": tempfile.mkdtemp(prefix="earth-pro-render-")})
        earth = subprocess.Popen(["google-earth-pro", "/root/camera-view2.kml"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(20)
        # A fresh profile can show the recovery dialog; Continue is on the
        # right-hand button, followed by the file trust dialog's Open button.
        click(830, 351)
        time.sleep(4)
        # Earth Pro may show its trust dialog after the first window paint.
        # Click the Open button repeatedly while the dialog is settling.
        for _ in range(4):
            click(495, 380)
            time.sleep(2)
        time.sleep(12)
        for window in subprocess.run(["xdotool", "search", "--name", "Start-up Tips"], env={**os.environ, **DISPLAY_ENV}, capture_output=True, text=True).stdout.split():
            subprocess.run(["xdotool", "windowclose", window], env={**os.environ, **DISPLAY_ENV})
        # Close any remaining transient trust dialog by clicking its Open button.
        for _ in range(3):
            click(495, 380)
            time.sleep(1)
        # Start from the world view and zoom into the tour area before taking
        # the first frame. Earth Pro needs several wheel ticks after a fresh
        # profile before local 3D tiles become visible.
        wheel(4, 8)
        time.sleep(10)
        time.sleep(3)

        previous_heading = 0.0
        previous_range = cameras[0]["range"]
        for index, camera in enumerate(cameras, 1):
            delta_heading = ((camera["heading"] - previous_heading + 180) % 360) - 180
            if abs(delta_heading) > 1:
                key("Right" if delta_heading > 0 else "Left")
                for _ in range(max(0, round(abs(delta_heading) / 4) - 1)):
                    key("Right" if delta_heading > 0 else "Left")
            delta_range = previous_range - camera["range"]
            if delta_range > 0:
                wheel(4, round(delta_range / 17))
            elif delta_range < 0:
                wheel(5, round(-delta_range / 17))
            time.sleep(1.5)

            raw = Path(tempfile.gettempdir()) / f"earth-pro-{index:02d}.png"
            subprocess.run(["import", "-display", ":150", "-window", "root", str(raw)], check=True)
            image = Image.open(raw).convert("RGB").crop((SCREEN_X, SCREEN_Y, SCREEN_X + FRAME_SIZE, SCREEN_Y + FRAME_SIZE))
            image.save(output_dir / f"image-{index:02d}.jpeg", format="JPEG", quality=90, optimize=True)
            previous_heading = camera["heading"]
            previous_range = camera["range"]
            print(f"{index}/55 {output_dir / f'image-{index:02d}.jpeg'}")
    finally:
        if earth is not None:
            earth.send_signal(signal.SIGTERM)
            try:
                earth.wait(timeout=5)
            except subprocess.TimeoutExpired:
                earth.kill()
        xvfb.terminate()
        xvfb.wait(timeout=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
