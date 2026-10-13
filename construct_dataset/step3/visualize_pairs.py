#!/usr/bin/env python3
"""Visualize valid pair transformations from pairs.jsonl."""

from __future__ import annotations

import argparse
import json
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def signed_angle_difference(target: float, source: float) -> float:
    return (target - source + 180.0) % 360.0 - 180.0


def load_values(path: Path) -> dict[str, np.ndarray]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {
        "dx": np.array([row["delta"]["x_m"] for row in rows]),
        "dy": np.array([row["delta"]["y_m"] for row in rows]),
        "dz": np.array([row["delta"]["z_m"] for row in rows]),
        "dtilt": np.array([row["target_camera"]["tilt_deg"] - row["source_camera"]["tilt_deg"] for row in rows]),
        "dheading": np.array([signed_angle_difference(row["target_camera"]["heading_deg"], row["source_camera"]["heading_deg"]) for row in rows]),
        "source_tilt": np.array([row["source_camera"]["tilt_deg"] for row in rows]),
        "target_tilt": np.array([row["target_camera"]["tilt_deg"] for row in rows]),
        "source_heading": np.array([row["source_camera"]["heading_deg"] for row in rows]),
        "target_heading": np.array([row["target_camera"]["heading_deg"] for row in rows]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=Path("/root/processed_paired_json/0000/pairs.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("/root/processed_paired_json/0000/visualizations"))
    parser.add_argument("--reference-pairs", type=Path, help="Previous pairs.jsonl for fixed-bin comparison")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    values = load_values(args.pairs)
    reference = load_values(args.reference_pairs) if args.reference_pairs else None
    manifest_path = args.pairs.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    limits = dict(zip(("dx", "dy", "dz", "dtilt", "dheading"), manifest.get("delta_limits", [40., 40., 40., 40., 180.])))

    labels = {
        "dx": "Delta X (m)", "dy": "Delta Y (m)", "dz": "Delta Z (m)",
        "dtilt": "Delta Tilt (deg)", "dheading": "Delta Heading (deg)",
    }
    colors = {"dx": "#2563eb", "dy": "#16a34a", "dz": "#dc2626", "dtilt": "#9333ea", "dheading": "#ea580c"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    for ax, key in zip(axes.flat, ["dx", "dy", "dz", "dtilt", "dheading"]):
        data = values[key]
        edges = np.linspace(-limits[key], limits[key], 25)
        if reference is not None:
            ax.hist(reference[key], bins=edges, histtype="step", color="#64748b", linewidth=1.5, label="Previous")
        ax.hist(data, bins=edges, color=colors[key], alpha=0.75, edgecolor="white", label="Current")
        ax.axhline(len(data)/24, color="#111827", linestyle="--", linewidth=1, label="Uniform target")
        ax.set_xlim(-limits[key], limits[key])
        ax.legend(fontsize=8)
        ax.axvline(0, color="#111827", linewidth=1)
        ax.set_title(f"{labels[key]}  (n={len(data)})")
        ax.set_xlabel(labels[key])
        ax.set_ylabel("Pair count")
        ax.grid(alpha=0.2)
    axes[1, 2].axis("off")
    axes[1, 2].text(.05, .7, "Fixed equal-width bins: 24\n"
                   f"XYZ: [-40, 40] m\nTilt: [-{limits['dtilt']:g}, {limits['dtilt']:g}] deg\nHeading: [-180, 180] deg\n"
                   "1000 unique unordered pairs\nTarget: 41 or 42 pairs per bin", fontsize=13,
                   transform=axes[1, 2].transAxes)
    fig.suptitle("Valid Pair Transformation Distributions", fontsize=16)
    fig.savefig(args.output_dir / "pair_delta_distributions.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    scatter = axes[0].scatter(values["dx"], values["dy"], c=values["dz"], cmap="coolwarm", s=9, alpha=0.7)
    axes[0].set_title("Horizontal displacement colored by Delta Z")
    axes[0].set_xlabel("Delta X (m)")
    axes[0].set_ylabel("Delta Y (m)")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].grid(alpha=0.2)
    fig.colorbar(scatter, ax=axes[0], label="Delta Z (m)")
    axes[1].scatter(values["dheading"], values["dtilt"], c=values["dz"], cmap="viridis", s=9, alpha=0.7)
    axes[1].set_title("Angular transformation colored by Delta Z")
    axes[1].set_xlabel("Delta Heading (deg, wrapped)")
    axes[1].set_ylabel("Delta Tilt (deg)")
    axes[1].grid(alpha=0.2)
    fig.savefig(args.output_dir / "pair_delta_scatter.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(values["dx"], values["dy"], values["dz"], c=values["dtilt"], cmap="plasma", s=8, alpha=0.7)
    ax.set_xlabel("Delta X (m)")
    ax.set_ylabel("Delta Y (m)")
    ax.set_zlabel("Delta Z (m)")
    ax.set_title("3D Camera-Center Displacement")
    fig.savefig(args.output_dir / "pair_xyz_3d.png", dpi=180)
    plt.close(fig)

    stats = {}
    for key in ["dx", "dy", "dz", "dtilt", "dheading"]:
        data = values[key]
        stats[key] = {"min": float(data.min()), "max": float(data.max()), "mean": float(data.mean()), "std": float(data.std()), "histograms": {}}
        for bins in (6, 12, 24):
            edges = np.linspace(-limits[key], limits[key], bins+1)
            counts, _ = np.histogram(data, bins=edges)
            hist = {"edges": edges.tolist(), "counts": counts.tolist(), "min_count": int(counts.min()),
                    "max_count": int(counts.max()), "cv": float(counts.std()/counts.mean())}
            if reference is not None:
                old_counts, _ = np.histogram(reference[key], bins=edges)
                hist["previous_counts"] = old_counts.tolist()
                hist["previous_cv"] = float(old_counts.std()/old_counts.mean())
            stats[key]["histograms"][str(bins)] = hist
    fig, axes = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    joint = {}
    for ax, (a, b) in zip(axes.flat, itertools.combinations(limits, 2)):
        counts, _, _ = np.histogram2d(values[a], values[b], bins=[np.linspace(-limits[a], limits[a], 5), np.linspace(-limits[b], limits[b], 5)])
        ax.imshow(counts.T, origin="lower", vmin=40, vmax=85, cmap="Blues", extent=(-limits[a], limits[a], -limits[b], limits[b]), aspect="auto")
        for i in range(4):
            for j in range(4):
                ax.text(-limits[a]+(i+.5)*limits[a]/2, -limits[b]+(j+.5)*limits[b]/2, str(int(counts[i,j])), ha="center", va="center", fontsize=9)
        ax.set_xlabel(labels[a])
        ax.set_ylabel(labels[b])
        joint[f"{a}_{b}"] = {"counts": counts.astype(int).tolist(), "min_count": int(counts.min()), "max_count": int(counts.max())}
    fig.suptitle("Pair coverage in every 2D projection (4 x 4 bins; ideal 62.5 per cell)")
    fig.savefig(args.output_dir / "pair_joint_coverage.png", dpi=150)
    plt.close(fig)
    stats["joint_coverage"] = joint
    (args.output_dir / "pair_delta_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote visualizations to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
