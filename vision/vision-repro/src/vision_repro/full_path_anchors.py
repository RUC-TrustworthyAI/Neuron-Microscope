"""Build all-layer top60 path anchors from the existing 50k input/score cache."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import time
import json
import numpy as np
import torch
from .all_neuron_examples import LAYERS, AllLayerCapture
from .five_variants import gaussian_weights, write_json, sha256
from .full_feature_viz import DEFAULT_SCAN, paths_for, previous_layer
from .models import load_inceptionv1


def topk_stable(values, k=60, negative=False):
    ordered = np.asarray(values) if negative else -np.asarray(values)
    threshold = np.partition(ordered, k-1)[k-1]
    candidates = np.flatnonzero(ordered <= threshold)
    return candidates[np.lexsort((candidates, ordered[candidates]))[:k]]


def build(output, scan=DEFAULT_SCAN, batch_size=24, request_chunk=16):
    output=Path(output);scan=Path(scan)
    output.mkdir(parents=True,exist_ok=True)
    if (output/'complete.json').exists():
        return json.loads((output/'complete.json').read_text())
    source=json.loads((scan/'protocol.json').read_text())
    assert source['images_scanned']==50000
    assert (scan/'complete.json').is_file()
    torch.set_num_threads(4)
    model,model_info=load_inceptionv1()
    assert source['model']['weights_sha256']==model_info['weights_sha256']
    capture=AllLayerCapture(model)
    inputs=np.load(scan/'inputs.npy',mmap_mode='r')
    assert inputs.shape==(50000,224,224,3)
    requests={};ranks={};locations={};sums={};counts={};calibrations={}
    started=time.monotonic()
    protocol={'created_at':datetime.now(timezone.utc).isoformat(),'model':model_info,
              'scan':str(scan),'scan_protocol_sha256':sha256(scan/'protocol.json'),
              'selection_sha256':sha256(scan/'selected_images.jsonl'),'images':50000,
              'top_k':60,'sigma':.12,'batch_size':batch_size,
              'definition':'Gaussian at natural target peak/minimum; previous catalog layer + current post-ReLU pooled vectors; target channel zeroed; each layer unit normalized; concatenation normalized; mean of top60 vectors then normalized',
              'previous_layers':{l:previous_layer(l) for l,_,_ in LAYERS},
              'conv2d0_previous':'input RGB in [0,1]',
              'ranking':'stable spatial max post-ReLU / spatial min signed pre-ReLU from cached scan',
              'calibration':'median absolute Gaussian signed-pole score at its natural peak',
              'degenerate':'Zero scale or zero anchor vector is flagged unusable; never silently replaced with random anchors',
              'source_sha256':sha256(__file__)}
    write_json(output/'protocol.json',protocol)
    for li,(layer,c,g) in enumerate(LAYERS):
        score=np.load(scan/f'{layer}_scores.npy',mmap_mode='r')
        locations[layer]=np.load(scan/f'{layer}_locations.npy',mmap_mode='r')
        selected=np.empty((c,2,60),np.int32)
        sums[layer]=np.zeros((c,2,(LAYERS[li-1][1] if li else 3)+c),np.float64)
        counts[layer]=np.zeros((c,2),np.int32)
        calibrations[layer]=np.zeros((c,2,60),np.float32)
        for ch in range(c):
            for pole in range(2):
                selected[ch,pole]=topk_stable(score[:,ch,pole],negative=bool(pole))
                for rank,index in enumerate(selected[ch,pole]):
                    requests.setdefault(int(index),{}).setdefault(layer,[]).append((ch,pole,rank))
        ranks[layer]=selected
        np.save(output/f'{layer}_indices.npy',selected)
        progress={'phase':'selecting top60','layer':layer,'elapsed_seconds':time.monotonic()-started}
        write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    ids=sorted(requests)
    with torch.inference_mode():
        for offset in range(0,len(ids),batch_size):
            batch=ids[offset:offset+batch_size]
            x=torch.from_numpy(np.array(inputs[batch])).to('cuda',dtype=torch.float32).permute(0,3,1,2)/255
            previous=[x]
            def collect(layer,post,pre):
                rows=[(bi,index,*r) for bi,index in enumerate(batch) for r in requests[index].get(layer,[])]
                grid=post.shape[-1]
                for lo in range(0,len(rows),request_chunk):
                    sub=rows[lo:lo+request_chunk]
                    bi=torch.tensor([r[0] for r in sub],device='cuda')
                    channels=torch.tensor([r[2] for r in sub],device='cuda')
                    poles=np.array([r[3] for r in sub]);chs=np.array([r[2] for r in sub])
                    loc=np.array([locations[layer][r[1],r[2],r[3]] for r in sub])
                    centers=torch.as_tensor(np.stack((loc//grid/(grid-1)-.5,loc%grid/(grid-1)-.5),1),device='cuda',dtype=torch.float32)
                    selected_post=post[bi]
                    vectors=paths_for(previous[0][bi],selected_post,channels,.12,centers).cpu().numpy()
                    field=torch.where(torch.as_tensor(poles,device='cuda')[:,None,None].bool(),pre[bi,channels],post[bi,channels])
                    values=(field*gaussian_weights(selected_post,.12,centers)).sum((-2,-1)).cpu().numpy()
                    np.add.at(sums[layer],(chs,poles),vectors)
                    np.add.at(counts[layer],(chs,poles),1)
                    calibrations[layer][chs,poles,[r[4] for r in sub]]=values
                previous[0]=post
            capture(x,collect,validate=offset==0)
            if offset%(batch_size*50)==0 or offset+batch_size>=len(ids):
                progress={'phase':'pooling anchor paths','images':min(offset+batch_size,len(ids)),
                          'total':len(ids),'elapsed_seconds':time.monotonic()-started}
                write_json(output/'progress.json',progress);print(json.dumps(progress),flush=True)
    capture.close()
    unavailable=[]
    for layer,c,g in LAYERS:
        assert (counts[layer]==60).all()
        anchors={'_protocol_sha256':sha256(output/'protocol.json')}
        for ch in range(c):
            for pi,pole in enumerate(('positive','negative')):
                vector=sums[layer][ch,pi]
                norm=np.linalg.norm(vector)
                if norm>0:vector=vector/norm
                scale=float(np.median(np.abs(calibrations[layer][ch,pi])))
                usable=bool(norm>0 and scale>0 and np.isfinite(vector).all() and np.isfinite(scale))
                if not usable:unavailable.append([layer,ch,pole])
                anchors[f'{ch}_{pole}']={'vector':torch.from_numpy(vector.astype(np.float32)),
                    'score_scale':scale,'count':60,'usable':usable,
                    'input_indices':torch.from_numpy(ranks[layer][ch,pi]),
                    'gaussian_scores':torch.from_numpy(calibrations[layer][ch,pi])}
        torch.save(anchors,output/f'{layer}.pt')
    result={'neurons':5804,'anchors':11608,'images':len(ids),'elapsed_seconds':time.monotonic()-started,
            'unusable':unavailable,'files':{l:sha256(output/f'{l}.pt') for l,_,_ in LAYERS}}
    write_json(output/'complete.json',result)
    print(json.dumps({'anchors_complete':str(output),'seconds':result['elapsed_seconds'],'unusable':len(unavailable)}),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--scan',type=Path,default=DEFAULT_SCAN);p.add_argument('--batch-size',type=int,default=24)
    a=p.parse_args();build(a.output,a.scan,a.batch_size)


if __name__=='__main__':main()
