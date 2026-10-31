"""Read existing nested ImageNet tar and build bounded, auditable path anchors.

Select 50 images per class (50k total), not the unknown author ordering and not
the full 1.28M negative scan. No downloads or extraction into the dataset tree.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from datetime import datetime, timezone
import hashlib
import heapq
import io
import json
from pathlib import Path
import tarfile
import time

import numpy as np
from PIL import Image
import torch

from .models import PROJECT_ROOT, load_inceptionv1
from .five_variants import CHANNELS, Mixed4cCapture, path_vector, unit_mean_anchor, gaussian_weights, write_json, sha256


class SliceReader(io.RawIOBase):
    def __init__(self,path,offset,length):
        self.file=open(path,'rb');self.offset=offset;self.length=length;self.position=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.position
    def seek(self,offset,whence=0):
        pos=offset if whence==0 else (self.position if whence==1 else self.length)+offset
        if pos<0:raise ValueError('negative seek')
        self.position=pos;return pos
    def read(self,size=-1):
        size=max(0,self.length-self.position) if size<0 else min(size,max(0,self.length-self.position))
        self.file.seek(self.offset+self.position);value=self.file.read(size);self.position+=len(value);return value
    def close(self):
        self.file.close();super().close()


def center224(image):
    image=image.convert('RGB');w,h=image.size
    size=(256,int(256*h/w)) if w<=h else (int(256*w/h),256)
    image=image.resize(size,Image.Resampling.BILINEAR)
    x,y=round((size[0]-224)/2),round((size[1]-224)/2)
    return np.asarray(image.crop((x,y,x+224,y+224))).copy()


def load_class(archive,entry,per_class,seed):
    index,name,offset,length=entry;images=[];records=[]
    rng=np.random.default_rng(np.random.SeedSequence([seed,index]))
    with SliceReader(archive,offset,length) as stream,tarfile.open(fileobj=stream,mode='r:') as inner:
        members=sorted((m for m in inner.getmembers() if m.isfile() and m.name.lower().endswith(('.jpg','.jpeg','.png'))),key=lambda m:m.name)
        picks=rng.choice(len(members),min(per_class,len(members)),replace=False)
        for pick in picks:
            member=members[int(pick)]
            with inner.extractfile(member) as f:payload=f.read()
            with Image.open(io.BytesIO(payload)) as image:crop=center224(image)
            images.append(crop)
            records.append({'class_tar':name,'image':member.name,'image_sha256':hashlib.sha256(payload).hexdigest()})
    return index,np.stack(images),records


def bounded_classes(archive,entries,per_class,seed,workers):
    with ThreadPoolExecutor(max_workers=workers) as pool:
        waiting=deque();iterator=iter(entries)
        def submit():
            try:entry=next(iterator)
            except StopIteration:return False
            waiting.append(pool.submit(load_class,archive,entry,per_class,seed));return True
        for _ in range(workers*2):submit()
        while waiting:
            result=waiting.popleft().result();submit();yield result


def build(output,archive,per_class=50,seed=20261026,workers=4,max_classes=None,batch_size=24):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    archive=Path(archive);marker=archive.with_suffix(archive.suffix+'.verified')
    if not marker.exists():raise ValueError('Existing verified archive marker required; no download performed')
    torch.set_num_threads(4);model,provenance=load_inceptionv1();capture=Mixed4cCapture(model)
    with tarfile.open(archive,'r:') as outer:
        entries=[(i,m.name,m.offset_data,m.size) for i,m in enumerate(sorted((m for m in outer.getmembers() if m.isfile() and m.name.endswith('.tar')),key=lambda m:m.name))]
    if len(entries)!=1000:raise ValueError('Expected 1000 class archives')
    if max_classes:entries=entries[:max_classes]
    protocol={'archive':str(archive),'archive_bytes':archive.stat().st_size,'verification_marker':marker.read_text(),'md5_recomputed':False,'sampling':'uniform without replacement, per class, sorted inner filenames; SeedSequence([seed,class_index])','seed':seed,'per_class':per_class,'classes':len(entries),'crop':'RGB /255; PIL bilinear short side256, center224','rank_positive':'max spatial post-ReLU','rank_negative':'min spatial signed pre-ReLU','anchor_size':60,'anchor_definition':'Gaussian at natural winning location -> per-layer L2 unit pooled mixed4b/mixed4c channel vectors; current target channel zeroed; concat unit; mean of top60 units then unit normalize','sigma':.12,'model':provenance,'source_sha256':{Path(__file__).name:sha256(__file__),'five_variants.py':sha256(Path(__file__).with_name('five_variants.py'))},'author_exact_anchor':False,'negative_full_dataset_scan':False,'created_at':datetime.now(timezone.utc).isoformat()}
    write_json(output/'protocol.json',protocol)
    heaps={f'{ch}_{pole}':[] for ch in CHANNELS for pole in ('positive','negative')}
    processed=0;serial=0;started=time.monotonic()
    with (output/'selected_images.jsonl').open('w') as manifest,torch.inference_mode():
        for class_index,arrays,records in bounded_classes(archive,entries,per_class,seed,workers):
            for record in records:manifest.write(json.dumps(record)+'\n')
            for start in range(0,len(arrays),batch_size):
                batch=arrays[start:start+batch_size]
                x=torch.from_numpy(batch).to('cuda',dtype=torch.float32).permute(0,3,1,2)/255
                previous,post,pre=capture(x)
                if processed==0 and not torch.allclose(post,pre.relu(),atol=1e-4,rtol=1e-5):raise AssertionError('Branch concatenation mismatch')
                for ch in CHANNELS:
                    for pole in ('positive','negative'):
                        field=post if pole=='positive' else pre
                        flat=field[:,ch].flatten(1)
                        value,index=flat.max(1) if pole=='positive' else flat.min(1)
                        h,w=field.shape[-2:]
                        centers=torch.stack((index.div(w,rounding_mode='floor')/(h-1)-.5,(index%w)/(w-1)-.5),1)
                        vectors=path_vector(previous,post,ch,.12,centers).cpu()
                        gaussian=(field[:,ch]*gaussian_weights(field,.12,centers)).sum((-2,-1)).cpu()
                        values=value.cpu().tolist();indices=index.cpu().tolist();heap=heaps[f'{ch}_{pole}']
                        for i,val in enumerate(values):
                            quality=val if pole=='positive' else -val
                            serial+=1
                            if len(heap)>=60 and quality<=heap[0][0]:continue
                            item={'record':records[start+i],'rank_score':val,'gaussian_score':float(gaussian[i]),'location':[indices[i]//w,indices[i]%w],'vector':vectors[i].clone(),'crop':batch[i].copy()}
                            entry=(quality,serial,item)
                            if len(heap)<60:heapq.heappush(heap,entry)
                            else:heapq.heapreplace(heap,entry)
                processed+=len(batch)
            if (class_index+1)%20==0 or class_index==0 or class_index==entries[-1][0]:
                progress={'classes':class_index+1,'images':processed,'elapsed_seconds':time.monotonic()-started,'images_per_second':processed/max(time.monotonic()-started,1),'total_images':len(entries)*per_class}
                write_json(output/'progress.json',progress);manifest.flush();print(json.dumps(progress),flush=True)
    anchors={};manifest_data={}
    for key,heap in heaps.items():
        selected=[x[2] for x in sorted(heap,reverse=True)]
        if len(selected)<60:raise ValueError(f'Need 60 anchors, only {len(selected)}')
        vectors=torch.stack([x['vector'] for x in selected])
        anchor=unit_mean_anchor(vectors)
        scale=float(np.median([abs(x['gaussian_score']) for x in selected]))
        if not scale>0:raise ValueError('Degenerate natural score scale')
        anchors[key]={'vector':anchor,'score_scale':scale,'count':len(selected),'vectors':vectors}
        manifest_data[key]={'score_scale':scale,'mapping':capture.mapping(int(key.split('_')[0])),'selected':[{k:v for k,v in x.items() if k not in ('vector','crop')} for x in selected]}
        # Original center224 inputs, not synthetic or website receptive-field crops.
        sheet=Image.new('RGB',(224*8,224*2))
        for i,item in enumerate(selected[:16]):sheet.paste(Image.fromarray(item['crop']),((i%8)*224,(i//8)*224))
        sheet.save(output/f'{key}_top16.jpg',quality=90)
    torch.save(anchors,output/'anchors.pt')
    write_json(output/'anchor_manifest.json',manifest_data)
    write_json(output/'complete.json',{'images':processed,'classes':len(entries),'elapsed_seconds':time.monotonic()-started,'anchors_sha256':sha256(output/'anchors.pt'),'selection_sha256':sha256(output/'selected_images.jsonl')})
    capture.close();print(json.dumps({'complete':str(output),'images':processed}),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,default=PROJECT_ROOT.parent/'source/imagenet/archives/ILSVRC2012_img_train.tar')
    p.add_argument('--per-class',type=int,default=50);p.add_argument('--seed',type=int,default=20261026)
    p.add_argument('--max-classes',type=int);p.add_argument('--workers',type=int,default=4);p.add_argument('--batch-size',type=int,default=24)
    a=p.parse_args();build(a.output,a.archive,a.per_class,a.seed,a.workers,a.max_classes,a.batch_size)


if __name__=='__main__':main()
