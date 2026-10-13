#!/usr/bin/env python3
"""Audit written KML, pair directions, coverage and rendered image files."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from construct_dataset.step2.render_satellite import parse_frame
from construct_dataset.step3.generate_pairs import LIMITS, WGS84_A, WGS84_E2, camera_basis, visibility, landmark_footprint_pixels


def reconstruct_pose(frame: dict, origin: dict) -> np.ndarray:
    lat = math.radians(origin['latitude'])
    n = WGS84_A / math.sqrt(1-WGS84_E2*math.sin(lat)**2)
    m = WGS84_A*(1-WGS84_E2)/(1-WGS84_E2*math.sin(lat)**2)**1.5
    east = math.radians(frame['longitude']-origin['longitude'])*n*math.cos(lat)
    north = math.radians(frame['latitude']-origin['latitude'])*m
    tilt, heading = math.radians(frame['tilt']), math.radians(frame['heading'])
    horizontal = frame['range']*math.sin(tilt)
    return np.array([east-horizontal*math.sin(heading), north-horizontal*math.cos(heading),
                     frame['altitude']+frame['range']*math.cos(tilt), frame['tilt'],
                     (frame['heading']+180)%360-180])


def validate(pairs_dir: Path, image_dir: Path) -> dict:
    manifest = json.loads((pairs_dir/'manifest.json').read_text())
    image_size = tuple(manifest.get('image_size', [512,512]))
    limits = np.array(manifest.get('delta_limits', [40.,40.,40.,40.,180.]))
    rows = [json.loads(line) for line in (pairs_dir/'pairs.jsonl').read_text().splitlines()]
    assert len(rows) == 1000
    count = manifest['candidate_count']
    assert count <= 100
    assert manifest['pair_count'] == len(rows)
    assert manifest['image_count'] == count
    files = sorted((pairs_dir/'kml').glob('candidate_*.kml'))
    assert len(files) == count
    assert {p.stem for p in image_dir.glob('candidate_*.jpeg')} == {p.stem for p in files}
    poses, projections, hashes = [], [], set()
    max_projection = 0.
    # Dense independent surface projection also checks the analytic sphere test.
    azimuth, elevation = np.meshgrid(np.linspace(0, 2*np.pi, 128, endpoint=False), np.linspace(-np.pi/2, np.pi/2, 65))
    sphere = manifest['landmark_radius_m']*np.stack([np.cos(elevation)*np.cos(azimuth), np.cos(elevation)*np.sin(azimuth), np.sin(elevation)], axis=-1).reshape(-1,3)
    sphere += [0., 0., manifest['landmark']['altitude']]
    clearances = []
    footprint_sizes = []
    for file in files:
        frame = parse_frame(file)
        pose = reconstruct_pose(frame, manifest['landmark'])
        if 'max_camera_height_m' in manifest:
            assert pose[2] <= manifest['max_camera_height_m']
            footprint = landmark_footprint_pixels(pose, manifest['landmark']['altitude'], manifest['landmark_radius_m'], frame['horiz_fov'])
            assert min(footprint) >= manifest['min_landmark_footprint_pixels']
            footprint_sizes.append(footprint)
        assert abs(frame['horiz_fov']-manifest['horizontal_fov_deg']) < 1e-9
        result = visibility(pose, manifest['landmark']['altitude'], manifest['landmark_radius_m'], frame['horiz_fov'], manifest['frame_margin_fraction'])
        assert result['visible'], file
        clearances.append(result['sphere_clearance_m'])
        forward, right, up = camera_basis(pose)
        relative = sphere-pose[:3]
        depth = relative @ forward
        assert depth.min() > 0
        ndc = np.column_stack([relative @ right, relative @ up])/(depth[:,None]*math.tan(math.radians(frame['horiz_fov']/2)))
        assert np.abs(ndc).max() <= 1-manifest['frame_margin_fraction']+1e-7
        max_projection = max(max_projection, float(np.abs(ndc).max()))
        projections.append(result['landmark_ndc'])
        poses.append(pose)
        image = image_dir/f'{file.stem}.jpeg'
        with Image.open(image) as im:
            assert im.size == image_size
            im.verify()
        hashes.add(hashlib.sha256(image.read_bytes()).hexdigest())
    assert len(hashes) == count, 'Duplicate rendered images'
    poses = np.array(poses)
    identities, pair_ids, used = set(), set(), set()
    deltas = []
    max_pose_error = 0.
    for row in rows:
        a,b = row['source_candidate'],row['target_candidate']
        assert a != b
        assert row['pair_id'] not in pair_ids
        pair_ids.add(row['pair_id'])
        identity = tuple(sorted((a,b)))
        assert identity not in identities, 'Repeated pair, possibly reversed'
        identities.add(identity)
        used.update((a,b))
        for side,index in [('source',a),('target',b)]:
            assert row[f'{side}_kml'] == f'kml/candidate_{index+1:03d}.kml'
            assert row[f'{side}_image'] == f'candidate_{index+1:03d}.jpeg'
            camera = row[f'{side}_camera']
            reported = np.array([camera[key] for key in ('x_m','y_m','z_m','tilt_deg','heading_deg')])
            error = float(abs(reported-poses[index]).max())
            assert error < 1e-5, (side,index,error)
            max_pose_error = max(max_pose_error,error)
        delta = poses[b]-poses[a]
        delta[4] = (delta[4]+180)%360-180
        reported_delta = np.array([row['delta'][key] for key in ('x_m','y_m','z_m')]+[row['delta_tilt_deg'],row['delta_heading_deg']])
        assert np.allclose(delta,reported_delta,rtol=0,atol=1e-5)
        assert np.all(abs(reported_delta) <= limits)
        deltas.append(reported_delta)
    assert used == set(range(count))
    deltas = np.array(deltas)
    for bins in (6,12,24):
        for dimension,limit in enumerate(limits):
            counts,_ = np.histogram(deltas[:,dimension], bins=np.linspace(-limit,limit,bins+1))
            assert counts.min() >= 1000//bins
            assert counts.max() <= math.ceil(1000/bins)
    joint_min, joint_max = 1000,0
    for a,b in itertools.combinations(range(5),2):
        counts,_,_ = np.histogram2d(deltas[:,a],deltas[:,b], bins=[np.linspace(-limits[a],limits[a],5),np.linspace(-limits[b],limits[b],5)])
        assert counts.min() >= 40 and counts.max() <= 85
        joint_min,joint_max = min(joint_min,int(counts.min())),max(joint_max,int(counts.max()))
    report = {'passed': True, 'pair_count': len(rows), 'unique_unordered_pairs': len(identities),
              'kml_count': count, 'image_count': len(hashes), 'all_images_used': True,
              'delta_limits': limits.tolist(),
              'camera_height_range_m': [float(poses[:,2].min()),float(poses[:,2].max())],
              'min_landmark_footprint_pixels_at_512': float(np.min(footprint_sizes)) if footprint_sizes else None,
              'min_landmark_footprint_pixels': float(np.min(np.array(footprint_sizes)*np.array(image_size)/512)) if footprint_sizes else None,
              'image_size': list(image_size),
              'max_kml_metadata_error': max_pose_error, 'all_landmark_spheres_visible': True,
              'landmark_radius_m': manifest['landmark_radius_m'], 'min_sphere_clearance_m': min(clearances),
              'max_landmark_sphere_abs_ndc': max_projection, 'marginal_24_bin_bounds': [41,42],
              'joint_4x4_observed_count_bounds': [joint_min,joint_max]}
    visualizations = pairs_dir/'visualizations'
    visualizations.mkdir(exist_ok=True)
    (visualizations/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    # Nine samples across the tilt range, with the first-key point indicated.
    sample = np.argsort(poses[:,3])[np.linspace(0,count-1,9).round().astype(int)]
    fig, axes = plt.subplots(3,3,figsize=(12,12),constrained_layout=True)
    for ax,index in zip(axes.flat,sample):
        with Image.open(image_dir/f'candidate_{index+1:03d}.jpeg') as im:
            ax.imshow(im)
        x,y = projections[index]
        ax.scatter([(x+1)*image_size[0]/2],[(1-y)*image_size[1]/2],s=140,facecolors='none',edgecolors='#ff3030',linewidths=1.5)
        ax.set_title(f'#{index+1:03d} | tilt {poses[index,3]:.1f} deg')
        ax.axis('off')
    fig.suptitle('Landmark visibility samples (red circle = first-key location)')
    fig.savefig(visualizations/'landmark_visibility_samples.png',dpi=150)
    plt.close(fig)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pairs-dir',type=Path,default=Path('/root/processed_paired_json/0000'))
    parser.add_argument('--image-dir',type=Path,default=Path('/root/processed_images/0000'))
    args = parser.parse_args()
    print(json.dumps(validate(args.pairs_dir,args.image_dir),indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
