"""CPU-only, immutable RF display export from the existing pretrained sample scan."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .site_export import public_metadata
from .receptive_field_crops import model_geometry, crop_box, sampled_heatmap

def compact(x):
    return json.dumps(x, ensure_ascii=False, separators=(',', ':')).encode()

def sha(x):
    return hashlib.sha256(x).hexdigest()

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('scan',type=Path);p.add_argument('output',type=Path);p.add_argument('source',type=Path)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--targets',nargs='*')
    a=p.parse_args(argv)
    if not 1<=a.workers<=4:p.error('--workers must be between 1 and 4')
    a.output.mkdir(parents=True,exist_ok=False)
    geometry=model_geometry(a.source)
    known={f'{l}:{c}' for l,fields in geometry.items() for c in range(len(fields))}
    if a.targets is not None and (not a.targets or set(a.targets)-known):
        p.error('--targets must list known layer:channel keys')
    inputs=np.load(a.scan/'inputs.npy',mmap_mode='r')
    protocol=json.loads((a.scan/'protocol.json').read_bytes())
    release=a.output.name;started=time.monotonic();created=datetime.now(timezone.utc).isoformat()
    targets=[(l,c) for l,fields in geometry.items() for c in range(len(fields))
             if not a.targets or f'{l}:{c}' in a.targets]
    maps={l:np.load(a.scan/f'{l}_maps.npy',mmap_mode='r') for l in {l for l,c in targets}}
    validated={}
    def save(key,raw):
        file=a.output/key;file.parent.mkdir(parents=True,exist_ok=True);file.write_bytes(raw)
        return {'key':key,'sha256':sha(raw),'bytes':len(raw)}
    def encode(target):
        l,ch=target;f=geometry[l][ch]
        node=json.loads((a.scan/f'web/neurons/{l}/{ch}.json').read_bytes())
        samples=[s for b in node['exampleBuckets'] for s in b['examples']]
        assert len(samples)==48
        cropped=f.size<224
        tile_pixels=112
        if not cropped:
            with Image.open(a.scan/f'web/atlases/{samples[0]["atlas"]}.webp') as original:
                tile_pixels=original.height
                if original.width != tile_pixels*8:
                    raise ValueError('Expected an eight-column source atlas')
        atlas=Image.new('RGB',(8*112,6*112)) if cropped else None
        heat=[];records=[]
        for cell,s in enumerate(samples):
            arr=np.asarray(inputs[s['index']]);expected=s['pixel_sha256']
            if s['index'] not in validated:
                assert sha(arr.tobytes())==expected
                validated[s['index']]=expected
            assert validated[s['index']]==expected
            raw=maps[l][ch,cell];neg=s['pole']=='negative';loc=s['mapLocation']
            extremum=float(raw.min() if neg else raw.max())
            assert np.isclose(extremum,s['activation'],rtol=2e-5,atol=1e-3)
            assert tuple(loc)==np.unravel_index(raw.argmin() if neg else raw.argmax(),raw.shape)
            requested,box=crop_box(f,loc);x0,y0,x1,y1=box
            strength=np.maximum(-raw if neg else raw,0)
            aligned=sampled_heatmap(strength,f,box,pixels=24)
            peak=float(aligned.max());q=np.rint(aligned/max(peak,1e-12)*255).astype('uint8')
            heat.append(q.tobytes())
            s.update(sourceHeatShape=list(raw.shape),heatShape=[24,24],heatOffset=cell*24*24,
                     heatPeak=float(strength.max()),cropHeatPeak=peak,cropBounds=box,
                     requestedRF=requested,activationPosition=[(f.start+loc[0]*f.jump-y0)/(y1-y0),
                                                              (f.start+loc[1]*f.jump-x0)/(x1-x0)])
            if cropped:
                tile=Image.fromarray(arr).crop(box).resize((112,112),Image.Resampling.BICUBIC)
                atlas.paste(tile,(cell%8*112,cell//8*112))
                s.update(tile=cell,atlasColumns=8,atlasRows=6)
            else:
                s.update(atlasUrl=f'/samples/atlases/{s["atlas"]}.webp',atlasColumns=8,atlasRows=1)
        if cropped:
            buf=io.BytesIO();atlas.save(buf,format='WEBP',lossless=True,method=1)
            raw=buf.getvalue();digest=sha(raw)
            with Image.open(io.BytesIO(raw)) as decoded:
                assert decoded.convert('RGB').tobytes()==atlas.tobytes()
            records.append(save(f'images/{digest}.webp',raw))
            for s in samples:s['atlasUrl']=f'/media/{digest}.webp'
        node.pop('heatPath',None)
        node.update(heatBase64=base64.b64encode(b''.join(heat)).decode(),
                    cropPolicy={'mode':'rf_fit','field':asdict(f),'maximum':224,'tilePixels':tile_pixels,
                                'heatPixels':24,'interpolation':'bicubic'},release=release)
        records.append(save(f'examples/{release}/nodes/{l}/{ch}.json',compact(node)))
        return f'{l}:{ch}',records,cropped
    files=[];nodes={};cropped_nodes=0
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for i,(key,records,cropped) in enumerate(pool.map(encode,targets),1):
            nodes[key]=1;files.extend(records);cropped_nodes+=cropped
            if i%100==0 or i==len(targets):
                print(json.dumps({'neurons':i,'total':len(targets),'croppedNeurons':cropped_nodes,
                                  'seconds':round(time.monotonic()-started,1)}),flush=True)
    summary={'kind':'examples','release':release,'snapshotAt':created,'neurons':len(nodes),
             'samples':len(nodes)*48,'croppedNeurons':cropped_nodes,'fullInputNeurons':len(nodes)-cropped_nodes,
             'model':public_metadata(protocol['model']),'scan':a.scan.name,'policy':'rf_fit','maxCrop':224,
             'modelSourceSha256':sha(a.source.read_bytes()),
             'implementationSha256':sha(Path(__file__).with_name('receptive_field_crops.py').read_bytes()),
             'scanProtocolSha256':sha((a.scan/'protocol.json').read_bytes()),
             'validatedInputImages':len(validated),'validation':'input hashes, extrema, locations, WebP round-trip'}
    files.append(save(f'examples/{release}/catalog.json',compact({**summary,'nodes':nodes})))
    files.append(save(f'examples/{release}/manifest.json',compact(summary)))
    (a.output/'upload-manifest.json').write_bytes(compact({'summary':summary,'objects':files}))
    (a.output/'complete.json').write_bytes(compact({**summary,'objects':len(files),'bytes':sum(x['bytes'] for x in files)}))
    print(json.dumps({'complete':True,'neurons':len(nodes),'bytes':sum(x['bytes'] for x in files),
                      'seconds':round(time.monotonic()-started,1)}),flush=True)

if __name__=='__main__':main()
