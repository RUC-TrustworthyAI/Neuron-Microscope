"""Make high-quality web previews while retaining the immutable lossless RF export."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image

def compact(v):return json.dumps(v,ensure_ascii=False,separators=(',',':')).encode()
def sha(b):return hashlib.sha256(b).hexdigest()

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args(argv);a.output.mkdir(parents=True,exist_ok=False)
    src=json.loads((a.source/'upload-manifest.json').read_bytes());release=a.output.name
    records=[];mapping={};audit=[];started=time.monotonic()
    def save(key,raw):
        f=a.output/key;f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(raw)
        return {'key':key,'bytes':len(raw),'sha256':sha(raw)}
    def encode(item):
        original=(a.source/item['key']).read_bytes();assert sha(original)==item['sha256']
        im=Image.open(io.BytesIO(original)).convert('RGB');before=np.asarray(im,dtype=np.float32)
        raw=original;quality='lossless';psnr=None
        for candidate in (95,98,100):
            b=io.BytesIO();im.save(b,format='WEBP',quality=candidate,method=4)
            after=np.asarray(Image.open(io.BytesIO(b.getvalue())).convert('RGB'),dtype=np.float32)
            # Check each displayed tile, not only an average over a 48-image atlas.
            delta=(before-after)**2
            mse=delta.reshape(6,112,8,112,3).mean(axis=(1,3,4))
            worst=float(10*np.log10(255**2/max(float(mse.max()),1e-12)))
            average=float(10*np.log10(255**2/max(float(delta.mean()),1e-12)))
            if worst>=30 and average>=36:
                if len(b.getvalue())<len(original):raw=b.getvalue();quality=candidate;psnr=worst
                break
        record=save(f'images/{sha(raw)}.webp',raw)
        return item['sha256'],record,{'source':item['sha256'],'sha256':record['sha256'],
                    'quality':quality,'minimum_tile_psnr_db':psnr,'bytes':len(raw)}
    images=[x for x in src['objects'] if x['key'].startswith('images/')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i,(old,r,check) in enumerate(pool.map(encode,images),1):
            mapping[old]=r['sha256'];records.append(r);audit.append(check)
            if i%200==0 or i==len(images):print(json.dumps({'images':i,'total':len(images),'seconds':round(time.monotonic()-started,1)}),flush=True)
    nodes={}
    for item in src['objects']:
        if '/nodes/' not in item['key']:continue
        original=(a.source/item['key']).read_bytes();assert sha(original)==item['sha256']
        n=json.loads(original);n['release']=release
        for b in n['exampleBuckets']:
            for s in b['examples']:
                if s['atlasUrl'].startswith('/media/'):
                    old=Path(s['atlasUrl']).stem;s['atlasUrl']=f'/media/{mapping[old]}.webp'
        n['cropPolicy']['displayEncoding']='WebP quality >=95, atlas PSNR >=36 dB and every tile >=30 dB; lossless fallback'
        nodes[f'{n["layer"]}:{n["channel"]}']=1
        records.append(save(f'examples/{release}/nodes/{n["layer"]}/{n["channel"]}.json',compact(n)))
    summary={**src['summary'],'release':release,'webEncodedAt':datetime.now(timezone.utc).isoformat(),
             'losslessSource':a.source.name,'displayEncoding':'WebP quality >=95; atlas >=36 dB and every tile >=30 dB or lossless'}
    records.append(save(f'examples/{release}/catalog.json',compact({**summary,'nodes':nodes})))
    records.append(save(f'examples/{release}/manifest.json',compact(summary)))
    (a.output/'encoding-audit.json').write_bytes(compact(audit))
    (a.output/'upload-manifest.json').write_bytes(compact({'summary':summary,'objects':records}))
    (a.output/'complete.json').write_bytes(compact({**summary,'objects':len(records),'bytes':sum(r['bytes'] for r in records)}))
    print(json.dumps({'complete':True,'bytes':sum(r['bytes'] for r in records),'seconds':round(time.monotonic()-started,1)}),flush=True)

if __name__=='__main__':main()
