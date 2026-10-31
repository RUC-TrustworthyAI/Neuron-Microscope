"""Two-pass, all-layer ImageNet retrieval with measured native-grid maps.

The fixed 50/class subset is shared with natural_examples. Full precision inputs,
scores and selected maps stay in the run; the website receives WebP atlases and
per-map uint8 display heatmaps. No data download or model training is performed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tarfile
import time

import numpy as np
from PIL import Image
import torch

from .models import PROJECT_ROOT, load_inceptionv1, preprocess
from .natural_anchors import bounded_classes
from .natural_examples import bucket_specs
from .five_variants import BRANCHES, sha256, write_json

from .catalog import LAYERS


class EndCapture(Exception):
    pass


class AllLayerCapture:
    """Exact pre-ReLU branch order, validated against each actual post output."""
    def __init__(self, model):
        self.model=model; self.handles=[]; self.values={}; self.callback=None
        modules=dict(model.named_modules()); self.mappings={}
        for layer,channels,grid in LAYERS:
            names=([f'{layer}_pre_relu_conv'] if layer.startswith('conv') else
                   [f'{layer}_{branch}_pre_relu_conv' for branch in BRANCHES])
            self.mappings[layer]=[{'node':name,'channel':ch} for name in names
                                  for ch in range(modules[name].out_channels)]
            assert len(self.mappings[layer])==channels
            for name in names:
                def save(module,inputs,result,name=name): self.values[name]=result
                self.handles.append(modules[name].register_forward_hook(save))
            def emit(module,inputs,post,layer=layer,names=names,grid=grid,channels=channels):
                parts=[self.values.pop(name) for name in names]
                pre=parts[0] if len(parts)==1 else torch.cat(parts,1)
                assert post.shape[1:]==(channels,grid,grid)
                if self.validate and not torch.equal(post,pre.relu()):
                    raise AssertionError(f'Pre/post mismatch: {layer}')
                self.callback(layer,post,pre)
                if layer==LAYERS[-1][0]: raise EndCapture()
            self.handles.append(modules[layer].register_forward_hook(emit))

    def __call__(self,images,callback,validate=False):
        self.values={}; self.callback=callback; self.validate=validate
        try: self.model(preprocess(images))
        except EndCapture: pass

    def close(self):
        for h in self.handles: h.remove()
        self.values={}; self.callback=None


def extrema(post,pre):
    pv,pi=post.flatten(2).max(2); nv,ni=pre.flatten(2).min(2)
    return torch.stack((pv,nv),-1),torch.stack((pi,ni),-1)


def selected_ranks(scores):
    """Stable ascending input index breaks equal activation ties."""
    n,c,poles=scores.shape
    assert poles==2
    specs=bucket_specs(n); selected=np.empty((c,48),np.int32)
    cursor=0
    orders={pole:np.argsort((-1 if pole=='positive' else 1)*scores[:,:,pi],axis=0,kind='stable')
            for pi,pole in enumerate(('positive','negative'))}
    for _,_,pole,ranks in specs:
        rank=np.asarray(list(ranks))-1
        selected[:,cursor:cursor+len(rank)]=orders[pole][rank].T
        cursor+=len(rank)
    return selected


def quantize_heat(raw,negative=False):
    strength=np.maximum(-raw if negative else raw,0)
    peak=float(strength.max())
    q=np.rint(strength/peak*255).astype(np.uint8) if peak>0 else np.zeros_like(raw,dtype=np.uint8)
    return q,peak


def build(output,archive,max_classes=None,batch_size=24,workers=4,seed=20261026):
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    archive=Path(archive); marker=archive.with_suffix(archive.suffix+'.verified')
    if not marker.exists(): raise ValueError('Verified existing archive required')
    with tarfile.open(archive,'r:') as outer:
        entries=[(i,m.name,m.offset_data,m.size) for i,m in enumerate(sorted(
            (m for m in outer.getmembers() if m.isfile() and m.name.endswith('.tar')),key=lambda m:m.name))]
    assert len(entries)==1000
    if max_classes: entries=entries[:max_classes]
    total=len(entries)*50; specs=bucket_specs(total)
    torch.set_num_threads(4); model,provenance=load_inceptionv1(); capture=AllLayerCapture(model)
    started=time.monotonic()
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),'seed':seed,'per_class':50,
        'classes':len(entries),'images_scanned':total,'full_imagenet_scan':False,
        'sampling':'sorted classes/filenames, uniform without replacement; SeedSequence([seed,class_index])',
        'preprocessing':'RGB /255, PIL bilinear short side256, center224; model RGB*255-117',
        'positive':'spatial maximum of post-ReLU','negative':'spatial minimum of signed pre-ReLU',
        'tie_break':'ascending stable subset index','archive':str(archive),
        'archive_verification_marker':marker.read_text(),'model':provenance,
        'layers':[{'name':l,'channels':c,'grid':g} for l,c,g in LAYERS],
        'mapping':capture.mappings,'heatmap':'native-grid float32 on DGX; uint8 per-image normalized positive/negative strength on website; not pixel attribution',
        'display_encoding':'224px crops, WebP quality82 atlas8x1; no resize; lossless input cache on DGX',
        'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__},
        'source_sha256':{p:sha256(Path(__file__).with_name(p)) for p in
                         ['all_neuron_examples.py','natural_anchors.py','natural_examples.py','models.py']}}
    write_json(output/'protocol.json',protocol)
    def progress(phase,**kw):
        value={'phase':phase,'elapsed_seconds':round(time.monotonic()-started,1),**kw}
        write_json(output/'progress.json',value);print(json.dumps(value),flush=True)
    inputs=np.lib.format.open_memmap(output/'inputs.npy',mode='w+',dtype=np.uint8,shape=(total,224,224,3))
    scores={}; locations={}; ranks={}; records=[]
    for l,c,g in LAYERS:
        scores[l]=np.lib.format.open_memmap(output/f'{l}_scores.npy',mode='w+',dtype=np.float32,shape=(total,c,2))
        locations[l]=np.lib.format.open_memmap(output/f'{l}_locations.npy',mode='w+',dtype=np.int16,shape=(total,c,2))
    processed=0
    with (output/'selected_images.jsonl').open('w') as manifest,torch.inference_mode():
        for ci,arrays,items in bounded_classes(archive,entries,50,seed,workers):
            for record in items: manifest.write(json.dumps(record)+'\n')
            records.extend(items)
            inputs[processed:processed+len(arrays)]=arrays
            for start in range(0,len(arrays),batch_size):
                batch=arrays[start:start+batch_size]; end=processed+len(batch)
                def collect(layer,post,pre):
                    vals,loc=extrema(post,pre)
                    scores[layer][processed:end]=vals.cpu().numpy()
                    locations[layer][processed:end]=loc.cpu().numpy()
                x=torch.from_numpy(batch).to('cuda',dtype=torch.float32).permute(0,3,1,2)/255
                capture(x,collect,validate=processed==0)
                processed=end
            if ci%20==0 or processed==total:
                manifest.flush(); progress('ranking scan',images=processed,total=total)
    assert processed==total
    inputs.flush()
    requests={}; raw_maps={}; heat={}; completed={}
    for l,c,g in LAYERS:
        scores[l].flush(); locations[l].flush()
        ranks[l]=selected_ranks(scores[l]); np.save(output/f'{l}_selected.npy',ranks[l])
        raw_maps[l]=np.lib.format.open_memmap(output/f'{l}_maps.npy',mode='w+',dtype=np.float32,shape=(c,48,g,g))
        heat[l]=np.empty((c,48,g,g),np.uint8); completed[l]=np.zeros((c,48),bool)
        for ch in range(c):
            for cell,index in enumerate(ranks[l][ch]):
                requests.setdefault(int(index),{}).setdefault(l,[]).append((ch,cell))
        progress('selecting ranks',layer=l,neurons=c)
    selected=sorted(requests); progress('map scan',images=0,total=total)
    # Preserve first-pass batch composition (50/class, including the final short
    # batch). CUDA convolution kernels can otherwise change numerical rounding.
    replay=[list(range(base+start,base+min(start+batch_size,50)))
            for base in range(0,total,50) for start in range(0,50,batch_size)]
    with torch.inference_mode():
        for batch_number,ids in enumerate(replay):
            needed={}
            for bi,index in enumerate(ids):
                for l,pairs in requests.get(index,{}).items():
                    needed.setdefault(l,[]).extend((bi,ch,cell,index) for ch,cell in pairs)
            def collect_maps(layer,post,pre):
                rows=needed.get(layer,[])
                if not rows:return
                bi=torch.tensor([r[0] for r in rows],device='cuda')
                ch=torch.tensor([r[1] for r in rows],device='cuda')
                values=pre[bi,ch].cpu().numpy()
                for row,raw in zip(rows,values):
                    _,channel,cell,index=row; negative=cell>=32
                    if not negative:raw=np.maximum(raw,0)
                    value=float(raw.min() if negative else raw.max())
                    reference=float(scores[layer][index,channel,int(negative)])
                    # Different replay batch shapes may alter FP32 convolution rounding.
                    if not np.isclose(value,reference,rtol=2e-5,atol=1e-3):
                        raise AssertionError((layer,channel,index,value,reference))
                    raw_maps[layer][channel,cell]=raw
                    heat[layer][channel,cell]=quantize_heat(raw,negative)[0]
                    completed[layer][channel,cell]=True
            x=torch.from_numpy(np.array(inputs[ids])).to('cuda',dtype=torch.float32).permute(0,3,1,2)/255
            capture(x,collect_maps)
            if batch_number%50==0 or ids[-1]+1==total:
                progress('map scan',images=ids[-1]+1,total=total)
    capture.close()
    for l,c,g in LAYERS:
        assert completed[l].all(); raw_maps[l].flush()
    web=output/'web'; (web/'atlases').mkdir(parents=True); (web/'neurons').mkdir(); (web/'heat').mkdir()
    atlas_index={index:(i//8,i%8) for i,index in enumerate(selected)}
    def encode_atlas(number):
        ids=selected[number*8:(number+1)*8]
        sheet=Image.new('RGB',(224*8,224))
        for cell,index in enumerate(ids): sheet.paste(Image.fromarray(inputs[index]),(cell*224,0))
        dest=web/'atlases'/f'{number}.webp'; sheet.save(dest,quality=82,method=4)
        return {'path':f'atlases/{number}.webp','sha256':sha256(dest),'bytes':dest.stat().st_size}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        assets=[]
        for i,result in enumerate(pool.map(encode_atlas,range((len(selected)+7)//8))):
            assets.append(result)
            if i%250==0:progress('encoding atlases',atlases=i+1,total=(len(selected)+7)//8)
    pixel_hashes={index:hashlib.sha256(inputs[index].tobytes()).hexdigest() for index in selected}
    for layer,c,g in LAYERS:
        (web/'neurons'/layer).mkdir(); (web/'heat'/layer).mkdir()
        for ch in range(c):
            buckets=[];cell=0
            for key,title,pole,ranklist in specs:
                examples=[]
                for rank in ranklist:
                    index=int(ranks[layer][ch,cell]);negative=pole=='negative'
                    loc=int(locations[layer][index,ch,int(negative)]); atlas,tile=atlas_index[index]
                    raw=raw_maps[layer][ch,cell]; q,peak=quantize_heat(raw,negative)
                    examples.append({'id':str(cell),'index':index,'rank':rank,'pole':pole,
                        'activation':float(scores[layer][index,ch,int(negative)]),
                        'location':[loc//g,loc%g], 'mapActivation':float(raw.min() if negative else raw.max()),
                        'mapLocation':list(map(int,np.unravel_index(raw.argmin() if negative else raw.argmax(),raw.shape))),
                        'heatOffset':cell*g*g,'heatShape':[g,g],'heatPeak':peak,
                        'atlas':atlas,'tile':tile,'label':Path(records[index]['class_tar']).stem,
                        'image':records[index]['image'],'pixel_sha256':pixel_hashes[index]})
                    cell+=1
                buckets.append({'id':key,'title':title,'pole':pole,'examples':examples})
            dest=web/'neurons'/layer/f'{ch}.json'
            dest.write_text(json.dumps({'layer':layer,'channel':ch,'mapping':capture.mappings[layer][ch],
                'heatPath':f'heat/{layer}/{ch}.bin','exampleBuckets':buckets},separators=(',',':')))
            hp=web/'heat'/layer/f'{ch}.bin';hp.write_bytes(heat[layer][ch].tobytes())
            assets.extend({'path':str(p.relative_to(web)),'sha256':sha256(p),'bytes':p.stat().st_size} for p in (dest,hp))
        progress('exporting neurons',layer=layer,neurons=c)
    selection_sha=sha256(output/'selected_images.jsonl')
    write_json(web/'protocol.json',protocol)
    write_json(web/'manifest.json',assets)
    complete={'images_scanned':total,'neurons':sum(c for _,c,_ in LAYERS),'samples':48*sum(c for _,c,_ in LAYERS),
        'unique_images':len(selected),'atlases':(len(selected)+7)//8,'atlas_columns':8,'atlas_rows':1,'selection_sha256':selection_sha,
        'manifest_sha256':sha256(web/'manifest.json'),'elapsed_seconds':time.monotonic()-started,
        'asset_bytes':sum(a['bytes'] for a in assets)}
    write_json(output/'complete.json',complete);write_json(web/'complete.json',complete)
    progress('complete',**{k:v for k,v in complete.items() if k!='elapsed_seconds'})


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--archive',type=Path,default=PROJECT_ROOT.parent/'source/imagenet/archives/ILSVRC2012_img_train.tar')
    p.add_argument('--max-classes',type=int);p.add_argument('--batch-size',type=int,default=24)
    p.add_argument('--workers',type=int,default=4);p.add_argument('--seed',type=int,default=20261026)
    a=p.parse_args();build(a.output,a.archive,a.max_classes,a.batch_size,a.workers,a.seed)


if __name__=='__main__':main()
