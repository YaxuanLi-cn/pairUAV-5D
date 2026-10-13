#!/usr/bin/env python3
"""Audit a completed tour batch and create totals, distributions and overview."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def summarize(root: Path, start: int, end: int):
    output=root/'batch_runs'/f'tours_{start:04d}_{end:04d}'
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    samples=[]
    values=[]
    for number in range(start,end+1):
        tour=f'{number:04d}'
        pairs=root/'processed_paired_json'/tour
        image_dir=root/'processed_images'/tour
        manifest=json.loads((pairs/'manifest.json').read_text())
        validation=json.loads((pairs/'visualizations/validation_report.json').read_text())
        data=[json.loads(line) for line in (pairs/'pairs.jsonl').read_text().splitlines()]
        images=sorted(image_dir.glob('candidate_*.jpeg'))
        assert validation['passed'] and manifest['pair_count']==len(data)==1000,tour
        assert 46<=len(images)==manifest['candidate_count']==manifest['image_count']<=100,tour
        assert manifest['tour_id']==tour,tour
        assert manifest['delta_limits']==[40.,40.,40.,25.,180.],tour
        assert manifest['minimum_tile_zoom']==20,tour
        assert manifest['image_size']==[1024,1024],tour
        assert manifest['render_quality']['jpeg_quality']==98,tour
        assert manifest['render_quality']['supersample']==2,tour
        assert manifest['actual_height_range_m'][1]<=210,tour
        identities={tuple(sorted((row['source_candidate'],row['target_candidate']))) for row in data}
        assert len(identities)==1000,tour
        matrix=np.array([[row['delta']['x_m'],row['delta']['y_m'],row['delta']['z_m'],row['delta_tilt_deg'],row['delta_heading_deg']] for row in data])
        for dimension,limit in enumerate(manifest['delta_limits']):
            assert np.all(np.abs(matrix[:,dimension])<=limit),tour
            histogram,_=np.histogram(matrix[:,dimension],bins=np.linspace(-limit,limit,25))
            assert histogram.min()>=41 and histogram.max()<=42,tour
        for path in images:
            with Image.open(path) as image:
                assert image.size==(1024,1024),path
                assert image.layer[0][1:3]==(1,1),path
                image.verify()
        values.append(matrix)
        rows.append({'tour_id':tour,'pair_count':1000,'image_count':len(images),'seed':manifest['seed'],
                     'sampling_mode':manifest.get('sampling_mode','uniform'),
                     'min_height_m':manifest['actual_height_range_m'][0],
                     'max_height_m':manifest['actual_height_range_m'][1],
                     'min_landmark_pixels':validation['min_landmark_footprint_pixels'],
                     'validated':True})
        # Same named camera for an easily navigable contact sheet.
        samples.append((tour,image_dir/'candidate_021.jpeg'))
    with (output/'tour_index.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    matrix=np.concatenate(values)
    labels=['Delta X (m)','Delta Y (m)','Delta Z (m)','Delta Tilt (deg)','Delta Heading (deg)']
    limits=[40,40,40,25,180]
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    histograms={}
    for axis,key,limit,data in zip(axes.flat,labels,limits,matrix.T):
        bins=np.linspace(-limit,limit,25)
        counts,_=np.histogram(data,bins=bins)
        histograms[key]={'counts':counts.tolist(),'edges':bins.tolist()}
        axis.hist(data,bins=bins,color='#2563eb',edgecolor='white')
        axis.axhline(len(matrix)/24,color='black',linestyle='--',linewidth=1)
        axis.set_xlabel(key);axis.set_ylabel('Pair count');axis.set_xlim(-limit,limit)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(.05,.6,f'{len(rows)} tours\n{len(matrix):,} pairs\n{sum(row["image_count"] for row in rows):,} images\n1024 x 1024 | zoom 20\nEach tour independently verified',fontsize=15)
    fig.suptitle(f'Tours {start:04d} - {end:04d}: aggregate pair differences')
    fig.savefig(output/'aggregate_distributions.png',dpi=150)
    plt.close(fig)
    # Split into four pages so that each landmark remains readable.
    contact_sheets=[]
    for page,offset in enumerate(range(0,len(samples),25),1):
        subset=samples[offset:offset+25]
        fig,axes=plt.subplots(5,5,figsize=(15,15),constrained_layout=True)
        for axis in axes.flat:axis.axis('off')
        for axis,(tour,path) in zip(axes.flat,subset):
            with Image.open(path) as image:
                axis.imshow(image)
            axis.set_title(f'Tour {tour} / candidate_021')
        fig.suptitle(f'Tour overview {subset[0][0]} - {subset[-1][0]}')
        name=f'overview_{page:02d}.png'
        fig.savefig(output/name,dpi=120)
        plt.close(fig)
        contact_sheets.append(name)
    summary={'passed':True,'tour_start':start,'tour_end':end,'tour_count':len(rows),'pair_count':len(matrix),
             'image_count':sum(row['image_count'] for row in rows),
             'image_count_per_tour':dict(Counter(row['image_count'] for row in rows)),
             'sampling_modes':dict(Counter(row['sampling_mode'] for row in rows)),
             'image_size':[1024,1024],
             'height_range_m':[min(row['min_height_m'] for row in rows),max(row['max_height_m'] for row in rows)],
             'min_landmark_pixels':min(row['min_landmark_pixels'] for row in rows),
             'all_marginal_24_bin_counts_within':[41,42],
             'histograms':histograms,'overview_images':contact_sheets}
    (output/'final_validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='histograms'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('/root'))
    parser.add_argument('--start',type=int,default=1)
    parser.add_argument('--end',type=int,default=100)
    args=parser.parse_args()
    summarize(args.root,args.start,args.end)
