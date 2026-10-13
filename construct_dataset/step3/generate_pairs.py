#!/usr/bin/env python3
"""Generate visible landmark views and 1000 uniformly balanced camera pairs.

The 24 fixed bins in each of dx/dy/dz/dtilt/dheading contain 41 or 42
pairs. Both directions are considered, but each unordered image pair is
used at most once. Candidate count grows only when selection is infeasible.
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

KML_NS = 'http://www.opengis.net/kml/2.2'
GX_NS = 'http://www.google.com/kml/ext/2.2'
WGS84_A = 6_378_137.0
WGS84_E2 = 6.6943799901413165e-3
DIMENSIONS = ('dx', 'dy', 'dz', 'dtilt', 'dheading')
LIMITS = np.array([40., 40., 40., 25., 180.])
ET.register_namespace('', KML_NS)
ET.register_namespace('gx', GX_NS)


def target_point(path: Path) -> np.ndarray:
    root = ET.parse(path).getroot()
    text = root.findtext(f'.//{{{KML_NS}}}Point/{{{KML_NS}}}coordinates')
    if not text:
        raise ValueError(f'{path}: no landmark Point coordinates')
    values = [float(v) for v in text.strip().split(',')]
    return np.array((values + [0.])[:3])


def geodetic_offset(origin: np.ndarray, east: float, north: float) -> tuple[float, float]:
    """Local tangent-plane offset, used identically when auditing KML poses."""
    lat = math.radians(origin[1])
    n = WGS84_A / math.sqrt(1 - WGS84_E2 * math.sin(lat)**2)
    m = WGS84_A * (1 - WGS84_E2) / (1 - WGS84_E2 * math.sin(lat)**2)**1.5
    return origin[0] + math.degrees(east / (n * math.cos(lat))), origin[1] + math.degrees(north / m)


def camera_basis(pose: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tilt, heading = np.radians(pose[3:5])
    forward = np.array([math.sin(heading)*math.sin(tilt), math.cos(heading)*math.sin(tilt), -math.cos(tilt)])
    right = np.array([math.cos(heading), -math.sin(heading), 0.])
    up = np.cross(right, forward)
    return forward, right, up


def visibility(pose: np.ndarray, landmark_altitude: float, radius: float,
               fov: float, margin: float) -> dict:
    """Require the entire landmark bounding sphere inside all frustum planes.

    Radius is a configurable geometric envelope around the first-key point;
    this tests field of view, not terrain/building occlusion.
    """
    forward, right, up = camera_basis(pose)
    rel = np.array([0., 0., landmark_altitude]) - pose[:3]
    depth = float(rel @ forward)
    tangent = math.tan(math.radians(fov / 2))
    half = math.atan((1 - margin) * tangent)
    clearance = min(depth * math.sin(half) - abs(float(rel @ axis)) * math.cos(half) for axis in (right, up))
    return {
        'visible': bool(depth > radius and clearance >= radius),
        'landmark_ndc': [float(rel @ right) / (depth*tangent), float(rel @ up) / (depth*tangent)] if depth > 0 else None,
        'sphere_clearance_m': float(clearance - radius),
    }


def landmark_footprint_pixels(pose: np.ndarray, altitude: float, radius: float, fov: float) -> list[float]:
    """Projected width/height of the landmark's horizontal footprint at 512px."""
    angle = np.linspace(0., 2*np.pi, 181)
    footprint = np.column_stack((radius*np.cos(angle), radius*np.sin(angle), np.full_like(angle, altitude)))
    forward, right, up = camera_basis(pose)
    relative = footprint-pose[:3]
    depth = relative @ forward
    if depth.min() <= 0:
        return [0., 0.]
    ndc = np.column_stack((relative @ right, relative @ up))/(depth[:,None]*math.tan(math.radians(fov/2)))
    return (np.ptp(ndc, axis=0)*256).tolist()


def sample_candidates(count: int, rng: np.random.Generator, altitude: float,
                      height: float, radius: float, fov: float, margin: float,
                      max_tilt: float = 25., extent: float = 30., min_footprint_pixels: float = 200.) -> np.ndarray:
    poses = []
    for _ in range(count):
        # Edge-rich tilt support prevents triangular pair-difference histograms.
        # Sample tilt first, so visibility rejection cannot erase extreme tilts.
        tilt = max_tilt * rng.beta(.5, .5)
        for _attempt in range(100_000):
            xyz = rng.uniform(-extent, extent, 3)
            xyz[2] = rng.uniform(height-30., height+30.)
            pose = np.r_[xyz, tilt, rng.uniform(-180., 180.)]
            if (visibility(pose, altitude, radius, fov, margin)['visible']
                    and min(landmark_footprint_pixels(pose, altitude, radius, fov)) >= min_footprint_pixels):
                poses.append(pose)
                break
        else:
            raise RuntimeError('No close-up view satisfies visibility and minimum landmark size. Adjust the angle span; do not automatically raise the camera.')
    return np.array(poses)


def sample_candidates_conditional(count: int, rng: np.random.Generator, altitude: float,
                                  height: float, radius: float, fov: float, margin: float,
                                  max_tilt: float = 25., extent: float = 30.,
                                  min_footprint_pixels: float = 200.) -> np.ndarray:
    """Sample feasible height boundaries when elevated landmarks erase Z tails.

    This changes the proposal distribution only. The original visibility,
    near-view scale, position envelope and pair constraints are still required.
    Each frustum side plane yields a linear lower bound on camera height.
    """
    poses=[]
    half=math.atan((1-margin)*math.tan(math.radians(fov/2)))
    ceiling=height+30.
    for index in range(count):
        tilt=max_tilt*rng.beta(.5,.5)
        choices=[]
        for _attempt in range(2000):
            xy=rng.uniform(-extent,extent,2)
            heading=rng.uniform(-180.,180.)
            forward,right,up=camera_basis(np.r_[xy,height,tilt,heading])
            normals=np.array([forward*math.sin(half)+sign*axis*math.cos(half)
                              for axis in (right,up) for sign in (-1,1)])
            if np.any(normals[:,2]>=0):
                raise ValueError('Conditional sampler requires all frustum planes to face down')
            minimum=max(height-30.,altitude+float(np.max((radius+normals[:,:2]@xy)/(-normals[:,2]))))
            if minimum<ceiling:
                choices.append((minimum,xy,heading))
            if len(choices)>=(32 if index%3==0 else 1):
                break
        if not choices:
            raise RuntimeError('No visible camera inside the unchanged near-view height limits')
        minimum,xy,heading=min(choices,key=lambda item:item[0])
        if index%3==0:
            z=minimum+(ceiling-minimum)*rng.uniform(0.,.06)
        elif index%3==1:
            z=rng.uniform(max(minimum,ceiling-2.),ceiling)
        else:
            z=rng.uniform(minimum,ceiling)
        pose=np.r_[xy,z,tilt,heading]
        if not visibility(pose,altitude,radius,fov,margin)['visible']:
            raise RuntimeError('Conditional camera failed the original visibility test')
        if min(landmark_footprint_pixels(pose,altitude,radius,fov))<min_footprint_pixels:
            raise RuntimeError('Conditional camera failed the original landmark scale test')
        poses.append(pose)
    return np.array(poses)


def pair_pool(poses: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = np.array(list(itertools.combinations(range(len(poses)), 2)))
    deltas = poses[indices[:, 1]] - poses[indices[:, 0]]
    deltas[:, 4] = (deltas[:, 4] + 180.) % 360. - 180.
    valid = (abs(deltas[:, :3]) <= 40.).all(axis=1)
    indices, deltas = indices[valid], deltas[valid]
    indices = np.concatenate((indices, indices[:, ::-1]))
    deltas = np.concatenate((deltas, -deltas))
    deltas[:, 4] = (deltas[:, 4] + 180.) % 360. - 180.
    return indices, deltas


def select_pairs(poses: np.ndarray, rng: np.random.Generator, time_limit: float, limits: np.ndarray = LIMITS) -> tuple | None:
    indices, deltas = pair_pool(poses)
    count, half = len(indices), len(indices)//2
    if half < 1000:
        print(f'  Only {half} valid unordered pairs', flush=True)
        return None
    norm = (deltas + limits) / (2*limits)
    rows, cols, lows, highs = [], [], [], []

    def constraint(columns, low, high):
        rows.extend([len(lows)] * len(columns))
        cols.extend(columns)
        lows.append(low)
        highs.append(high)

    constraint(np.arange(count), 1000, 1000)
    # Nested fixed grids prevent a solution that is balanced only at one scale.
    for bins in (24, 12, 6):
        buckets = np.clip((norm*bins).astype(int), 0, bins-1)
        for dimension in range(5):
            for bucket in range(bins):
                members = np.flatnonzero(buckets[:, dimension] == bucket)
                if len(members) < 1000//bins:
                    print(f'  Insufficient support: {DIMENSIONS[dimension]}, {bins} bins, bucket {bucket}', flush=True)
                    return None
                constraint(members, 1000//bins, math.ceil(1000/bins))
    # No reverse-pair duplication to inflate the requested 1000 combinations.
    for pair in range(half):
        constraint([pair, pair+half], 0, 1)
    # Spread every 2-D projection too, within a stated feasible tolerance.
    coarse = np.clip((norm*4).astype(int), 0, 3)
    for a, b in itertools.combinations(range(5), 2):
        buckets = coarse[:, a]*4 + coarse[:, b]
        for bucket in range(16):
            constraint(np.flatnonzero(buckets == bucket), 40, 85)
    # All generated images must participate, not simply inflate the manifest.
    for image in range(len(poses)):
        constraint(np.flatnonzero((indices == image).any(axis=1)), 1, np.inf)

    matrix = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(len(lows), count)).tocsc()
    result = milp(rng.uniform(0, .001, count), integrality=np.ones(count), bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix, lows, highs),
                  options={'time_limit': time_limit, 'mip_rel_gap': 1.})
    print(f'  Solver: {result.message}', flush=True)
    if result.x is None:
        return None
    chosen = np.rint(result.x).astype(int)
    values = matrix @ chosen
    if not (np.all(values >= np.array(lows)-1e-6) and np.all(values <= np.array(highs)+1e-6)):
        return None
    selected = np.flatnonzero(chosen)
    return indices[selected], deltas[selected]


def adjusted_kml(landmark: np.ndarray, pose: np.ndarray, name: str, fov: float) -> ET.ElementTree:
    root = ET.Element(f'{{{KML_NS}}}kml')
    document = ET.SubElement(root, f'{{{KML_NS}}}Document')
    ET.SubElement(document, f'{{{KML_NS}}}name').text = name
    pm = ET.SubElement(document, f'{{{KML_NS}}}Placemark')
    ET.SubElement(pm, f'{{{KML_NS}}}name').text = 'first-key-landmark'
    ET.SubElement(pm, f'{{{KML_NS}}}visibility').text = '0'
    point = ET.SubElement(pm, f'{{{KML_NS}}}Point')
    ET.SubElement(point, f'{{{KML_NS}}}altitudeMode').text = 'relativeToGround'
    ET.SubElement(point, f'{{{KML_NS}}}coordinates').text = ','.join(f'{v:.12f}' for v in landmark)
    # A ground-level LookAt makes the camera height equal to range*cos(tilt),
    # with no hidden change of reference plane when view targets move.
    forward, _, _ = camera_basis(pose)
    distance = pose[2] / -forward[2]
    center = pose[:3] + distance*forward
    lon, lat = geodetic_offset(landmark, center[0], center[1])
    look = ET.SubElement(pm, f'{{{KML_NS}}}LookAt')
    for key, value in {'longitude': lon, 'latitude': lat, 'altitude': 0., 'heading': pose[4],
                       'tilt': pose[3], 'range': distance}.items():
        ET.SubElement(look, f'{{{KML_NS}}}{key}').text = f'{value:.12f}'
    ET.SubElement(look, f'{{{KML_NS}}}altitudeMode').text = 'relativeToGround'
    ET.SubElement(look, f'{{{GX_NS}}}horizFov').text = f'{fov:.12f}'
    return ET.ElementTree(root)


def distribution_report(deltas: np.ndarray, limits: np.ndarray = LIMITS) -> dict:
    report = {}
    for j, name in enumerate(DIMENSIONS):
        counts, edges = np.histogram(deltas[:, j], bins=np.linspace(-limits[j], limits[j], 25))
        report[name] = {'bin_edges': edges.tolist(), 'counts': counts.tolist(),
                        'min_count': int(counts.min()), 'max_count': int(counts.max()),
                        'cv': float(counts.std()/counts.mean())}
    return report


def write_outputs(output: Path, tour_id: str, first_key: Path, landmark: np.ndarray,
                  poses: np.ndarray, indices: np.ndarray, deltas: np.ndarray, settings: dict, attempts: list):
    output.mkdir(parents=True, exist_ok=True)
    # Selection must succeed before replacing an existing dataset.
    with tempfile.TemporaryDirectory(prefix='.uniform-', dir=output.parent) as temp:
        staging = Path(temp)
        (staging/'kml').mkdir()
        cameras = []
        for index, pose in enumerate(poses):
            name = f'candidate_{index+1:03d}'
            adjusted_kml(landmark, pose, name, settings['horizontal_fov_deg']).write(staging/'kml'/f'{name}.kml', encoding='utf-8', xml_declaration=True)
            cameras.append({'x_m': float(pose[0]), 'y_m': float(pose[1]), 'z_m': float(pose[2]),
                            'tilt_deg': float(pose[3]), 'heading_deg': float(pose[4]),
                            'range_m': float(pose[2]/math.cos(math.radians(pose[3])))})
        with (staging/'pairs.jsonl').open('w') as handle:
            for pair_id, ((source, target), delta) in enumerate(zip(indices, deltas), 1):
                row = {'pair_id': f'{tour_id}_{pair_id:06d}', 'tour_id': tour_id, 'first_key': str(first_key),
                       'source_candidate': int(source), 'target_candidate': int(target),
                       'source_kml': f'kml/candidate_{source+1:03d}.kml', 'target_kml': f'kml/candidate_{target+1:03d}.kml',
                       'source_camera': cameras[source], 'target_camera': cameras[target],
                       'delta': dict(zip(('x_m','y_m','z_m'), map(float,delta[:3]))),
                       'delta_tilt_deg': float(delta[3]), 'delta_heading_deg': float(delta[4])}
                handle.write(json.dumps(row, sort_keys=True)+'\n')
        visible = [visibility(p, landmark[2], settings['landmark_radius_m'], settings['horizontal_fov_deg'], settings['frame_margin_fraction']) for p in poses]
        manifest = {'tour_id': tour_id, 'candidate_count': len(poses), 'pair_count': len(indices), 'image_count': 0,
                    **settings, 'attempts': attempts, 'coordinate_frame': 'ENU metres; XY origin at first-key landmark; Z relative to local ground',
                    'landmark': dict(zip(('longitude','latitude','altitude'), map(float,landmark))),
                    'visibility': {'all_visible': all(v['visible'] for v in visible),
                                   'min_sphere_clearance_m': min(v['sphere_clearance_m'] for v in visible),
                                   'candidates': visible},
                    'uniformity': {'marginal_bins': 24, 'marginal_count_bounds': [41,42],
                                   'nested_bins': [12,6], 'joint_bins_per_axis': 4, 'joint_count_bounds': [40,85],
                                   'dimensions': distribution_report(deltas, np.array(settings['delta_limits']))}}
        (staging/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
        kml_dir = output/'kml'
        kml_dir.mkdir(exist_ok=True)
        for old in kml_dir.glob('candidate_*.kml'):
            old.unlink()
        for path in (staging/'kml').iterdir():
            path.replace(kml_dir/path.name)
        for name in ('pairs.jsonl','manifest.json'):
            (staging/name).replace(output/name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tour-id', default='0000')
    parser.add_argument('--candidate-count', type=int, default=70, help='Starting image count; grows by 10 when necessary')
    parser.add_argument('--max-candidate-count', type=int, default=100)
    parser.add_argument('--seed-version', default='step3-v4-closeup')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sampling-mode', choices=('auto','uniform','conditional'), default='auto')
    parser.add_argument('--first-key', type=Path, default=Path('/root/first-key/0000.kml'))
    parser.add_argument('--split-dir', type=Path, default=Path('/root/split_kml/0000'), help='Retained for CLI compatibility; all poses derive from first-key')
    parser.add_argument('--output-dir', type=Path, default=Path('/root/processed_paired_json/0000'))
    parser.add_argument('--landmark-radius', type=float, default=60., help='Radius of the sphere that must remain inside every image (m)')
    parser.add_argument('--camera-height', type=float, default=180., help='Mean height above ground; candidates vary +/-30 m')
    parser.add_argument('--max-camera-height', type=float, default=210., help='Hard upper limit; never expanded by the solver')
    parser.add_argument('--horizontal-fov', type=float, default=70.)
    parser.add_argument('--max-tilt', type=float, default=25., help='Absolute tilt range and signed pair tilt limit')
    parser.add_argument('--min-landmark-pixels', type=float, default=200., help='Minimum width AND height of the 60m-radius footprint at 512px')
    parser.add_argument('--solver-seconds', type=float, default=45.)
    args = parser.parse_args()
    if not 46 <= args.candidate_count <= args.max_candidate_count <= 100:
        parser.error('Require 46 <= candidate-count <= max-candidate-count <= 100')
    if args.landmark_radius <= 0 or args.solver_seconds <= 0:
        parser.error('landmark-radius and solver-seconds must be positive')
    landmark = target_point(args.first_key)
    height = args.camera_height
    if height+30 > args.max_camera_height:
        parser.error('camera-height + 30 exceeds the hard max-camera-height')
    if not 0 < args.horizontal_fov < 180 or not 0 < args.max_tilt < 90 or args.max_tilt+args.horizontal_fov/2 >= 89:
        parser.error('Require a ground-intersecting field of view and positive tilt range')
    if not 0 < args.min_landmark_pixels < 512:
        parser.error('min-landmark-pixels must be between 0 and 512')
    limits = np.array([40., 40., 40., args.max_tilt, 180.])
    if height <= max(30., landmark[2]+args.landmark_radius+30.):
        parser.error('camera-height must keep all cameras above the landmark envelope')
    seed = args.seed
    attempts = []
    modes = ('uniform','conditional') if args.sampling_mode=='auto' else (args.sampling_mode,)
    for mode in modes:
        sampler = sample_candidates if mode=='uniform' else sample_candidates_conditional
        print(f'Sampling mode: {mode}', flush=True)
        poses = sampler(args.max_candidate_count, np.random.default_rng(seed), landmark[2], height,
                        args.landmark_radius, args.horizontal_fov, .03, args.max_tilt, 30., args.min_landmark_pixels)
        sizes = list(range(args.candidate_count, args.max_candidate_count+1, 10))
        if sizes[-1] != args.max_candidate_count:
            sizes.append(args.max_candidate_count)
        for size in sizes:
            print(f'Trying {size} visible images for 1000 distinct pairs', flush=True)
            result = select_pairs(poses[:size], np.random.default_rng(seed), args.solver_seconds, limits)
            attempts.append({'sampling_mode': mode, 'candidate_count': size, 'feasible_solution_found': result is not None})
            if result is not None:
                indices, deltas = result
                settings = {'sampling_mode': mode, 'seed_version': args.seed_version, 'landmark_radius_m': args.landmark_radius,
                            'camera_height_m': height, 'max_camera_height_m': args.max_camera_height,
                            'horizontal_fov_deg': args.horizontal_fov, 'frame_margin_fraction': .03,
                            'max_tilt_deg': args.max_tilt, 'delta_limits': limits.tolist(), 'seed': seed,
                            'min_landmark_footprint_pixels': args.min_landmark_pixels,
                            'actual_height_range_m': [float(poses[:size,2].min()), float(poses[:size,2].max())],
                            'landmark_footprint_pixels': [landmark_footprint_pixels(p, landmark[2], args.landmark_radius, args.horizontal_fov) for p in poses[:size]]}
                write_outputs(args.output_dir, args.tour_id, args.first_key, landmark, poses[:size], indices, deltas, settings, attempts)
                print(f'Generated {size} KMLs and {len(indices)} unique pairs; all five 24-bin histograms have 41–42 pairs per bin.')
                return 0
    raise RuntimeError('No verified uniform solution found within the image/time limits; existing outputs were preserved. Try a different --seed or a longer --solver-seconds; height and landmark size will not be relaxed.')


if __name__ == '__main__':
    raise SystemExit(main())
