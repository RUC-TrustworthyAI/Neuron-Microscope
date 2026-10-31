"""Our fixed ImageNet subset rankings and actual activation maps, not web samples."""
import argparse
from datetime import datetime, timezone
import hashlib
import heapq
import json
from pathlib import Path
import tarfile
import time

import numpy as np
from PIL import Image, features
import torch

from .models import PROJECT_ROOT, load_inceptionv1
from .natural_anchors import bounded_classes
from .five_variants import CHANNELS, Mixed4cCapture, sha256, write_json


def bucket_specs(count):
    if count < 504:
        raise ValueError('At least 504 samples required for the declared ranking windows')
    return [('top16', 'Top 16', 'positive', range(1, 17)),
            ('p999', 'Around 99.9%', 'positive', range(47, 55)),
            ('p99', 'Around 99%', 'positive', range(497, 505)),
            ('neg16', 'Most negative', 'negative', range(1, 17))]


def retain(heap, item, capacity, negative=False):
    quality = -item['activation'] if negative else item['activation']
    entry = (quality, -item['index'], item)
    if len(heap) < capacity:
        heapq.heappush(heap, entry)
    elif entry[:2] > heap[0][:2]:
        heapq.heapreplace(heap, entry)


def build(output, archive, per_class=50, seed=20261026, workers=4, max_classes=None, batch_size=24):
    output=Path(output); output.mkdir(parents=True, exist_ok=False)
    archive=Path(archive); marker=archive.with_suffix(archive.suffix+'.verified')
    if not marker.exists(): raise ValueError('Existing verified dataset required; no downloads')
    torch.set_num_threads(4)
    model, provenance=load_inceptionv1(); capture=Mixed4cCapture(model)
    with tarfile.open(archive, 'r:') as outer:
        entries=[(i,m.name,m.offset_data,m.size) for i,m in enumerate(sorted(
            (m for m in outer.getmembers() if m.isfile() and m.name.endswith('.tar')), key=lambda m:m.name))]
    if len(entries)!=1000: raise ValueError('Expected 1000 ImageNet class archives')
    if max_classes: entries=entries[:max_classes]
    specs=bucket_specs(len(entries)*per_class)
    protocol={'created_at':datetime.now(timezone.utc).isoformat(), 'seed':seed,
        'per_class':per_class, 'classes':len(entries), 'full_imagenet_scan':False,
        'archive':str(archive), 'archive_bytes':archive.stat().st_size,
        'verification_marker':marker.read_text(), 'md5_recomputed':False,
        'sampling':'sorted classes and filenames; uniform without replacement; SeedSequence([seed,class_index])',
        'crop':'RGB /255; PIL bilinear short side256; center224',
        'model':provenance, 'positive':'mixed4c post-ReLU spatial maximum',
        'negative':'signed pre-ReLU spatial minimum, exact branch concatenation',
        'heatmap':'Actual 14x14 response on center224 input; positive max(post,0); negative max(-pre,0). Per-image normalized only for display. Not pixel attribution.',
        'tie_break':'ascending stable subset index',
        'source_sha256':{p:sha256(Path(__file__).with_name(p)) for p in
                         ['natural_examples.py','natural_anchors.py','five_variants.py','models.py']},
        'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__}}
    write_json(output/'protocol.json',protocol)
    heaps={(ch,pole):[] for ch in CHANNELS for pole in ('positive','negative')}
    all_scores=[]; all_locations=[]; processed=0; started=time.monotonic()
    with (output/'selected_images.jsonl').open('w') as manifest, torch.inference_mode():
        for class_index, arrays, records in bounded_classes(archive,entries,per_class,seed,workers):
            for record in records: manifest.write(json.dumps(record)+'\n')
            for start in range(0,len(arrays),batch_size):
                batch=arrays[start:start+batch_size]
                x=torch.from_numpy(batch).to('cuda',dtype=torch.float32).permute(0,3,1,2)/255
                _,post,pre=capture(x)
                if not torch.allclose(post,pre.relu(),atol=1e-4,rtol=1e-5):
                    raise AssertionError('Signed branch mapping mismatch')
                post_maps=post[:,CHANNELS].cpu().numpy(); pre_maps=pre[:,CHANNELS].cpu().numpy()
                scores=np.empty((len(batch),len(CHANNELS),2),np.float32)
                locations=np.empty_like(scores,dtype=np.int16)
                for ci,ch in enumerate(CHANNELS):
                    for pi,pole in enumerate(('positive','negative')):
                        maps=(post_maps if pole=='positive' else pre_maps)[:,ci]
                        flat=maps.reshape(len(batch),-1)
                        loc=flat.argmax(1) if pole=='positive' else flat.argmin(1)
                        vals=flat[np.arange(len(batch)),loc]
                        scores[:,ci,pi]=vals; locations[:,ci,pi]=loc
                        heap=heaps[ch,pole]; cap=504 if pole=='positive' else 16
                        for i,val in enumerate(vals):
                            index=processed+i; quality=float(val) if pole=='positive' else -float(val)
                            if len(heap)>=cap and (quality,-index)<=heap[0][:2]: continue
                            item={'index':index, **records[start+i], 'activation':float(val),
                                  'location':[int(loc[i]//maps.shape[-1]),int(loc[i]%maps.shape[-1])],
                                  'crop':batch[i].copy(), 'map':maps[i].copy()}
                            retain(heap,item,cap,pole=='negative')
                all_scores.append(scores); all_locations.append(locations); processed+=len(batch)
            if (class_index+1)%50==0 or class_index==0 or class_index==entries[-1][0]:
                progress={'images':processed,'classes':class_index+1,'elapsed_seconds':time.monotonic()-started}
                write_json(output/'progress.json',progress); manifest.flush(); print(json.dumps(progress),flush=True)
    np.savez_compressed(output/'scan_scores.npz',scores=np.concatenate(all_scores),
                        locations=np.concatenate(all_locations),channels=CHANNELS,poles=['positive','negative'])
    (output/'images').mkdir(); neurons={}; saved=set()
    suffix='webp' if features.check('webp') else 'png'
    for ch in CHANNELS:
        ranked={pole:[e[2] for e in sorted(heaps[ch,pole],reverse=True)] for pole in ('positive','negative')}
        buckets=[]
        for key,title,pole,ranks in specs:
            examples=[]
            for rank in ranks:
                item=ranked[pole][rank-1]; rel=f'images/{item["index"]}.{suffix}'
                if rel not in saved:
                    im=Image.fromarray(item['crop'])
                    im.save(output/rel,**({'lossless':True,'method':4} if suffix=='webp' else {})); saved.add(rel)
                examples.append({k:v for k,v in item.items() if k not in ('crop','map')} | {
                    'id':f'{ch}-{pole}-{rank}', 'rank':rank, 'pole':pole, 'path':rel,
                    'label':Path(item['class_tar']).stem, 'heatmap':item['map'].tolist(),
                    'heatShape':list(item['map'].shape), 'pixel_sha256':hashlib.sha256(item['crop'].tobytes()).hexdigest(),
                    'asset_sha256':sha256(output/rel)})
            buckets.append({'id':key,'title':title,'pole':pole,'examples':examples})
        neurons[f'mixed4c:{ch}']={'layer':'mixed4c','channel':ch,'mapping':capture.mapping(ch),'exampleBuckets':buckets}
    selection_sha=sha256(output/'selected_images.jsonl')
    result={'neurons':neurons,'protocol':protocol,'images_scanned':processed,'selection_sha256':selection_sha,
            'asset_encoding':'lossless WebP' if suffix=='webp' else 'lossless PNG',
            'unique_images':len(saved),'elapsed_seconds':time.monotonic()-started}
    write_json(output/'examples.json',result)
    write_json(output/'complete.json',{'images_scanned':processed,'examples_sha256':sha256(output/'examples.json'),
        'selection_sha256':selection_sha,'scan_scores_sha256':sha256(output/'scan_scores.npz'),
        'elapsed_seconds':time.monotonic()-started})
    capture.close(); print(json.dumps({'complete':str(output),'images':processed,'unique':len(saved)}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,default=PROJECT_ROOT.parent/'source/imagenet/archives/ILSVRC2012_img_train.tar')
    p.add_argument('--max-classes',type=int); p.add_argument('--per-class',type=int,default=50)
    p.add_argument('--seed',type=int,default=20261026); p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args(); build(a.output,a.archive,a.per_class,a.seed,a.workers,a.max_classes,a.batch_size)


if __name__=='__main__': main()
