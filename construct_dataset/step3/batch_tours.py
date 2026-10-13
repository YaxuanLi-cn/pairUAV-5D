#!/usr/bin/env python3
"""Resume-capable batch generation, rendering and validation for tour ranges."""
from __future__ import annotations

import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(path)


class Batch:
    def __init__(self, args):
        self.args=args
        self.root=args.root.resolve()
        self.work=self.root/'batch_runs'/f'tours_{args.start:04d}_{args.end:04d}'
        self.work.mkdir(parents=True,exist_ok=True)
        (self.work/'logs').mkdir(exist_ok=True)
        self.mutex=threading.Lock()
        self.state={}
        self.started=now()
        self.env={**os.environ,'PYTHONUNBUFFERED':'1','MPLBACKEND':'Agg','OMP_NUM_THREADS':'1',
                  'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1'}
        self.ids=[f'{i:04d}' for i in range(args.start,args.end+1)]
        for tour in self.ids:
            status=self.work/f'{tour}.json'
            self.state[tour]=json.loads(status.read_text()) if status.exists() else {'tour_id':tour,'status':'pending'}
        self.save_summary()

    def save_summary(self):
        counts={status:sum(v['status']==status for v in self.state.values()) for status in ('pending','running','complete','failed')}
        summary={'started_at':self.started,'updated_at':now(),'tour_start':self.args.start,'tour_end':self.args.end,
                 'workers':self.args.workers,'counts':counts,'total_pairs':sum(v.get('pair_count',0) for v in self.state.values() if v['status']=='complete'),
                 'total_images':sum(v.get('image_count',0) for v in self.state.values() if v['status']=='complete'),
                 'failed_tours':[t for t,v in self.state.items() if v['status']=='failed'],'tours':list(self.state.values())}
        atomic_json(self.work/'summary.json',summary)

    def status(self,tour,**fields):
        with self.mutex:
            self.state[tour].update(fields,updated_at=now())
            atomic_json(self.work/f'{tour}.json',self.state[tour])
            self.save_summary()
        if 'stage' in fields or fields.get('status') in ('complete','failed'):
            print(f"[{now()}] {tour}: {fields}",flush=True)

    def command(self,tour,stage,module,arguments,timeout):
        self.status(tour,status='running',stage=stage)
        command=[sys.executable,'-m',f'construct_dataset.step3.{module}',*map(str,arguments)]
        with (self.work/'logs'/f'{tour}.log').open('a') as log:
            log.write(f'\n[{now()}] {json.dumps(command)}\n');log.flush()
            try:
                result=subprocess.run(command,cwd=self.root,env=self.env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
                return result.returncode==0
            except subprocess.TimeoutExpired:
                log.write(f'\nTIMEOUT after {timeout} seconds\n')
                return False

    def output_dirs(self,tour):
        return self.root/'processed_paired_json'/tour,self.root/'processed_images'/tour

    def generated(self,tour):
        pairs,_=self.output_dirs(tour)
        try:
            m=json.loads((pairs/'manifest.json').read_text())
            return (m['tour_id']==tour and m['pair_count']==1000 and 46<=m['candidate_count']<=100
                    and m.get('seed_version')=='step3-v4-closeup'
                    and m.get('delta_limits')==[40.,40.,40.,25.,180.]
                    and m.get('max_camera_height_m')==210.
                    and len(list((pairs/'kml').glob('candidate_*.kml')))==m['candidate_count'])
        except (OSError,KeyError,ValueError):
            return False

    def rendered(self,tour):
        pairs,images=self.output_dirs(tour)
        try:
            m=json.loads((pairs/'manifest.json').read_text())
            return (self.generated(tour) and m['image_count']==m['candidate_count'] and m.get('image_size')==[1024,1024]
                    and m.get('render_quality',{}).get('supersample')==2
                    and len(list(images.glob('candidate_*.jpeg')))==m['candidate_count'])
        except (OSError,KeyError,ValueError):
            return False

    def process(self,tour):
        start=time.monotonic()
        first=self.root/'first-key'/f'{tour}.kml'
        pairs,images=self.output_dirs(tour)
        try:
            if not first.exists():raise FileNotFoundError(first)
            prior=self.state[tour]
            if prior['status']=='complete' and self.rendered(tour) and (pairs/'visualizations/validation_report.json').exists():
                print(f'[{now()}] {tour}: already complete; skipping',flush=True)
                return
            if not self.generated(tour):
                success=False
                for seed in range(42,42+self.args.seed_attempts):
                    self.status(tour,seed=seed)
                    arguments=['--tour-id',tour,'--first-key',first,'--output-dir',pairs,'--seed',seed,
                               '--seed-version','step3-v4-closeup','--candidate-count',70,'--max-candidate-count',100,
                               '--solver-seconds',45,'--landmark-radius',60,'--camera-height',180,'--max-camera-height',210,
                               '--horizontal-fov',70,'--max-tilt',25,'--min-landmark-pixels',200]
                    if self.command(tour,f'generate(seed={seed})','generate_pairs',arguments,300):
                        success=True;break
                if not success:raise RuntimeError('No valid pair selection found within the unchanged close-up constraints and seed attempts')
            if self.args.generate_only:
                self.status(tour,status='pending',stage='generated')
                return
            if not self.rendered(tour):
                success=False
                for attempt in range(1,4):
                    arguments=['--pairs-dir',pairs,'--output-dir',images,'--cache-dir',self.args.cache_dir,
                               '--min-tile-zoom',20,'--image-size',1024,'--supersample',2,'--jpeg-quality',98]
                    if self.command(tour,f'render(attempt={attempt})','render_images',arguments,3600):
                        success=True;break
                if not success:raise RuntimeError('Rendering failed after three attempts; inspect the per-tour log')
            if not self.command(tour,'visualize','visualize_pairs',['--pairs',pairs/'pairs.jsonl','--output-dir',pairs/'visualizations'],180):
                raise RuntimeError('Visualization failed')
            if not self.command(tour,'validate','validate_pairs',['--pairs-dir',pairs,'--image-dir',images],180):
                raise RuntimeError('Validation failed')
            manifest=json.loads((pairs/'manifest.json').read_text())
            validation=json.loads((pairs/'visualizations/validation_report.json').read_text())
            if not validation['passed']:raise RuntimeError('Validation report did not pass')
            self.status(tour,status='complete',stage='complete',pair_count=manifest['pair_count'],image_count=manifest['image_count'],
                        seed=manifest['seed'],elapsed_seconds=round(time.monotonic()-start,1),
                        height_range_m=validation['camera_height_range_m'],min_landmark_pixels=validation['min_landmark_footprint_pixels'],
                        first_key_sha256=hashlib.sha256(first.read_bytes()).hexdigest(),error=None)
        except Exception as exc:
            self.status(tour,status='failed',error=str(exc),elapsed_seconds=round(time.monotonic()-start,1))

    def run(self):
        with (self.work/'batch.lock').open('w') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.args.workers) as pool:
                futures=[pool.submit(self.process,tour) for tour in self.ids]
                for future in concurrent.futures.as_completed(futures):future.result()
        self.save_summary()
        print((self.work/'summary.json').read_text(),flush=True)
        return 2 if any(v['status']=='failed' for v in self.state.values()) else 0


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--start',type=int,default=1)
    parser.add_argument('--end',type=int,default=100)
    parser.add_argument('--workers',type=int,default=12)
    parser.add_argument('--seed-attempts',type=int,default=16)
    parser.add_argument('--root',type=Path,default=Path('/root'))
    parser.add_argument('--cache-dir',type=Path,default=Path('/root/.cache/construct_dataset/google_tiles'))
    parser.add_argument('--generate-only',action='store_true')
    args=parser.parse_args()
    if not 1<=args.start<=args.end or args.workers<1 or args.seed_attempts<1:parser.error('Invalid tour range or worker/attempt count')
    return Batch(args).run()


if __name__=='__main__':
    raise SystemExit(main())
