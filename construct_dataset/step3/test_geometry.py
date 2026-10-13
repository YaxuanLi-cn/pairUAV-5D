"""Regression checks for pair direction, KML geometry and shifted rendering."""
import io
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from construct_dataset.step2 import render_satellite as renderer
from construct_dataset.step3.generate_pairs import adjusted_kml, pair_pool, visibility, landmark_footprint_pixels, sample_candidates_conditional
from construct_dataset.step3.validate_pairs import reconstruct_pose


class GeometryTests(unittest.TestCase):
    def test_kml_round_trip_keeps_camera_when_lookat_moves(self):
        landmark = np.array([-113.4689682002154,53.5214914000673,11.23922684951166])
        origin = dict(zip(('longitude','latitude','altitude'),landmark))
        with tempfile.TemporaryDirectory() as temp:
            for tilt,heading in [(0,0),(39.9,175),(20,-135)]:
                pose = np.array([28,-22,432,tilt,heading])
                path = Path(temp)/'view.kml'
                adjusted_kml(landmark,pose,'view',90).write(path)
                actual = reconstruct_pose(renderer.parse_frame(path),origin)
                np.testing.assert_allclose(actual,pose,rtol=0,atol=1e-5)

    def test_pair_pool_retains_directions_and_wraps_heading(self):
        poses = np.array([[0,0,200,0,179],[10,-20,220,40,-179],[100,100,300,0,0]])
        pairs,deltas = pair_pool(poses)
        np.testing.assert_array_equal(pairs,[[0,1],[1,0]])
        np.testing.assert_allclose(deltas,[[10,-20,20,40,2],[-10,20,-20,-40,-2]])

    def test_visibility_rejects_clipped_building_even_with_point_inside(self):
        pose = np.array([0,0,100,40,0])
        self.assertTrue(visibility(pose,0,1,90,0)['visible'])
        self.assertFalse(visibility(pose,0,20,90,0)['visible'])

    def test_ground_footprint_respects_lookat_altitude(self):
        frame = dict(longitude=0,latitude=0,altitude=20,range=100,heading=0,tilt=0,horiz_fov=60)
        points = renderer.camera_ground_points(frame,(512,512))
        self.assertAlmostEqual(max(p[0] for p in points),120*math.tan(math.pi/6))

    def test_landmark_pixel_constraint_rejects_previous_distant_scale(self):
        close = landmark_footprint_pixels(np.array([0,0,180,0,0]),11.24,60,70)
        distant = landmark_footprint_pixels(np.array([0,0,432,0,0]),11.24,60,90)
        self.assertGreater(min(close),200)
        self.assertLess(min(distant),200)

    def test_renderer_refuses_to_reduce_requested_tile_detail(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root/'frames'
            frames.mkdir()
            adjusted_kml(np.array([0.,0.,0.]),np.array([0.,0.,1000.,0.,0.]),'wide',90).write(frames/'wide.kml')
            with self.assertRaisesRegex(ValueError,'minimum tile resolution'):
                renderer.render(frames,root/'images',root/'cache',(512,512),per_frame_mosaic=True,min_tile_zoom=20)

    def test_elevated_landmark_sampling_keeps_original_near_view_constraints(self):
        poses=sample_candidates_conditional(100,np.random.default_rng(45),68.515176891598,180,60,70,.03)
        self.assertGreaterEqual(poses[:,2].min(),150)
        self.assertLessEqual(poses[:,2].max(),210)
        self.assertTrue(np.all(np.abs(poses[:,:2])<=30))
        self.assertTrue(np.all((poses[:,3]>=0)&(poses[:,3]<=25)))
        for pose in poses:
            self.assertTrue(visibility(pose,68.515176891598,60,70,.03)['visible'])
        _,deltas=pair_pool(poses)
        counts,_=np.histogram(deltas[:,2],bins=np.linspace(-40,40,25))
        self.assertGreaterEqual(counts.min(),41,'Conditional proposal must cover the formerly empty Z tails')

    def test_tile_cache_preserves_original_bytes_and_ignores_lossy_legacy(self):
        buffer = io.BytesIO()
        Image.new('RGB',(256,256),(31,124,212)).save(buffer,format='JPEG',quality=77)
        original = buffer.getvalue()
        with tempfile.TemporaryDirectory() as temp:
            cache = Path(temp)
            (cache/'20_1_2.jpg').write_bytes(b'legacy recompressed bytes')
            with patch.object(renderer.urllib.request,'urlopen',return_value=io.BytesIO(original)) as download:
                renderer.download_tile(1,2,20,cache)
                renderer.download_tile(1,2,20,cache)
            self.assertEqual(download.call_count,1)
            self.assertEqual((cache/'original/20_1_2.jpg').read_bytes(),original)

    def test_high_resolution_render_preserves_footprint_and_file_resolution(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            frames=root/'frames'
            frames.mkdir()
            adjusted_kml(np.array([0.,0.,0.]),np.array([0.,0.,10.,12.,15.]),'view',60).write(frames/'view.kml')
            frame=renderer.parse_frame(frames/'view.kml')
            np.testing.assert_allclose(renderer.camera_ground_points(frame,(32,32)),renderer.camera_ground_points(frame,(64,64)))
            mosaic=Image.new('RGB',(1024,1024),(30,80,120))
            with patch.object(renderer,'download_mosaic',return_value=(mosaic,512.,512.)):
                renderer.render(frames,root/'images',root/'cache',(64,64),supersample=2,jpeg_quality=98)
            with Image.open(root/'images/view.jpeg') as im:
                self.assertEqual(im.size,(64,64))
                self.assertEqual(im.layer[0][1:3],(1,1))  # no chroma subsampling

    def test_render_uses_each_frames_geographic_center(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            frames = root/'frames'
            frames.mkdir()
            landmark = np.array([0.,0.,0.])
            for index,x in enumerate([0.,10.]):
                adjusted_kml(landmark,np.array([x,0.,10.,0.,0.]),str(index),60).write(frames/f'{index}.kml')
            gradient = np.broadcast_to((np.arange(1024)//4).astype(np.uint8)[None,:,None],(1024,1024,3)).copy()
            mosaic = Image.fromarray(gradient)
            with patch.object(renderer,'download_mosaic',return_value=(mosaic,512.,512.)):
                renderer.render(frames,root/'images',root/'cache',(32,32))
            first = np.asarray(Image.open(root/'images/0.jpeg')).mean()
            second = np.asarray(Image.open(root/'images/1.jpeg')).mean()
            self.assertGreater(second-first,10.,'Shifted KMLs must not sample an identical tile center')


if __name__ == '__main__':
    unittest.main()
