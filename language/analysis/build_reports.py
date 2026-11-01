"""Batch original-layout reports from a saved dense BF16 capture; no model or LLM."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import argparse, hashlib, html, json, os, re, time, threading
import numpy as np
import build_report as core

K = 30
PROFILE = {
    "top_count":10, "bottom_count":8, "display_seed":42,
    "context_prefix_tokens":512,
    "bands":[{"key":key,"source_key":source,"count":5} for key,source in [
        ("p99_99","p99.99–100"),("p99_9","p99.9–99.99"),("p99","p99–99.9"),
        ("p95","p95–99"),("p90","p90–95"),("p0_01","p0–0.01"),("p0_1","p0.01–0.1")]],
    "quantiles":["p0_01","p0_1","p1","p5","p10","p25","p50","p75","p90","p95","p99","p99_9","p99_99"],
    "windows":{"top":{"max_tokens":55,"after":10,"before":44},
               "bottom":{"max_tokens":35,"after":10},"band":{"max_tokens":34,"after":10}},
    "band_window_extreme":True,
}
KEYS = ("top_tokens","bottom_tokens","top_documents","bottom_documents")
HIGH = np.array([True,False,True,False])
SIGN = np.ones(65536,dtype=np.uint8)
SIGN[(np.arange(65536)&32767)!=0] = 2
SIGN[((np.arange(65536)&32768)!=0)&((np.arange(65536)&32767)!=0)] = 0

def progress(out, **data):
    data["utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
    tmp=out/"progress.json.tmp"
    core.write(tmp,data);tmp.replace(out/"progress.json")

def read_record(record, n_units):
    raw=np.load(record["activation"],mmap_mode="r",allow_pickle=False)
    if raw.dtype!=np.uint16 or raw.shape!=(record["n"],n_units,3):
        raise ValueError("Capture shape/dtype mismatch: "+record["activation"])
    return raw.transpose(1,0,2).copy(order="C")


DEVICES=[]
THREAD=threading.local()

def pool_for(workers):
    def initialize(assignment):THREAD.device=next(assignment)
    return ThreadPoolExecutor(max_workers=workers,initializer=initialize,
                              initargs=(iter(DEVICES),)) if DEVICES else ThreadPoolExecutor(max_workers=workers)

def raw_record(record,n_units):
    raw=np.load(record["activation"],allow_pickle=False)
    if raw.dtype!=np.uint16 or raw.shape!=(record["n"],n_units,3):
        raise ValueError("Capture shape/dtype mismatch")
    return raw

def gpu_record(record,n_units,lookup=None):
    """Exact integer array operations; no model is loaded or evaluated."""
    import torch
    device=THREAD.device
    raw=raw_record(record,n_units)
    x=torch.from_numpy(raw).to(device).to(torch.int64)
    if lookup is not None:
        if not hasattr(THREAD,"lookup"):
            THREAD.lookup=torch.from_numpy(lookup).to(device).reshape(-1)
        ids=x[:,:,2]+torch.arange(n_units,device=device)[None,:]*65536
        bins=THREAD.lookup[ids].T.contiguous()
        counts=torch.bincount((bins.to(torch.int64)+torch.arange(n_units,device=device)[:,None]*14).reshape(-1),
                              minlength=n_units*14).reshape(n_units,14)
        return bins.cpu().numpy(),counts.cpu().numpy().astype(np.uint64)
    if bool(((x&0x7f80)==0x7f80).any()):raise ValueError("Nonfinite capture")
    g=x[:,:,0];v=x[:,:,1]
    gs=torch.where((g&32767)==0,1,torch.where((g&32768)!=0,0,2))
    vs=torch.where((v&32767)==0,1,torch.where((v&32768)!=0,0,2))
    quad=torch.bincount((gs*3+vs+torch.arange(n_units,device=device)[None,:]*9).reshape(-1),
                       minlength=n_units*9).reshape(n_units,3,3).cpu().numpy().astype(np.uint64)
    products=((x[:,:,2].to(torch.int32)<<16).view(torch.float32)).T.contiguous()
    x.add_(torch.arange(n_units*3,device=device).reshape(1,n_units,3)*65536)
    hist=torch.bincount(x.reshape(-1),minlength=n_units*3*65536).reshape(n_units,3,65536).cpu().numpy().astype(np.uint64)
    ids,scores=gpu_extremes(products,record)
    return hist,quad,ids,scores

def gpu_extremes(products,record):
    import torch
    n_units,n=products.shape;device=products.device
    ids=torch.full((n_units,4,K),np.iinfo(np.int64).max,dtype=torch.int64,device=device)
    scores=torch.zeros((n_units,4,K),dtype=torch.float32,device=device)
    off=np.load(record["offsets"],allow_pickle=False).astype(np.int64);docs=len(off)-1
    index=torch.repeat_interleave(torch.arange(docs,device=device),
                                 torch.from_numpy(np.diff(off)).to(device))[None,:].expand(n_units,-1)
    positions=torch.arange(n,device=device)[None,:].expand(n_units,-1)
    for slot,high in ((0,True),(1,False)):
        # Stable sort keeps the earliest native position for equal values, including signed zero.
        ranked=torch.argsort(products,dim=1,descending=high,stable=True)[:,:min(K,n)]
        ids[:,slot,:ranked.shape[1]]=ranked+record["start"]
        scores[:,slot,:ranked.shape[1]]=products.gather(1,ranked)
        ds=torch.full((n_units,docs),-float("inf") if high else float("inf"),device=device)
        ds.scatter_reduce_(1,index,products,reduce="amax" if high else "amin",include_self=True)
        matches=products==ds.gather(1,index)
        dp=torch.full((n_units,docs),n,device=device,dtype=torch.int64)
        dp.scatter_reduce_(1,index,torch.where(matches,positions,n),reduce="amin",include_self=True)
        chosen=torch.argsort(ds,dim=1,descending=high,stable=True)[:,:min(K,docs)]
        at=dp.gather(1,chosen)
        ids[:,slot+2,:at.shape[1]]=at+record["start"]
        scores[:,slot+2,:at.shape[1]]=products.gather(1,at)
    return ids.cpu().numpy(),scores.cpu().numpy()

def extreme_products(products,record):
    n_units=len(products)
    ids=np.full((n_units,4,K),np.iinfo(np.int64).max,dtype=np.int64)
    scores=np.zeros((n_units,4,K),dtype=np.float32)
    off=np.load(record["offsets"],allow_pickle=False).astype(np.int64)
    for col,product in enumerate(products):
        for high,slot in ((True,0),(False,1)):
            selected=core.best(product,K,high)
            ids[col,slot,:len(selected)]=selected+record["start"]
            scores[col,slot,:len(selected)]=product[selected]
            doc_scores=(np.maximum if high else np.minimum).reduceat(product,off[:-1])
            docs=core.best(doc_scores,K,high)
            positions=np.array([int(off[j])+int((np.argmax if high else np.argmin)(product[off[j]:off[j+1]])) for j in docs],dtype=np.int64)
            ids[col,slot+2,:len(positions)]=positions+record["start"]
            scores[col,slot+2,:len(positions)]=product[positions]
    return ids,scores

def scan_record(record, n_units):
    if DEVICES:return gpu_record(record,n_units)
    raw=read_record(record,n_units)
    if np.any((raw&0x7f80)==0x7f80):raise ValueError("Nonfinite capture")
    hist=np.zeros((n_units,3,65536),dtype=np.uint64)
    quadrants=np.zeros((n_units,3,3),dtype=np.uint64)
    for col,bits in enumerate(raw):
        for channel in range(3):
            hist[col,channel]=np.bincount(bits[:,channel],minlength=65536)
        quadrants[col]=np.bincount(SIGN[bits[:,0]]*3+SIGN[bits[:,1]],minlength=9).reshape(3,3)
    ids,scores=extreme_products(core.floats(raw[:,:,2]),record)
    return hist,quadrants,ids,scores

def ordered_map(pool, fn, records, workers):
    # Bounded submission: never queue thousands of 300MB histogram results.
    for start in range(0,len(records),workers):
        yield from pool.map(fn,records[start:start+workers])

def first_pass(records,n_units,out,workers):
    hist=np.zeros((n_units,3,65536),dtype=np.uint64)
    quad=np.zeros((n_units,3,3),dtype=np.uint64)
    best_ids=np.full((n_units,4,K),np.iinfo(np.int64).max,dtype=np.int64)
    best_scores=np.where(HIGH[None,:,None],-np.inf,np.inf)*np.ones((n_units,4,K))
    began=time.monotonic();total=sum(r["n"] for r in records)
    with pool_for(workers) as pool:
        fn=lambda record:scan_record(record,n_units)
        for index,(h,q,ids,scores) in enumerate(ordered_map(pool,fn,records,workers)):
            hist+=h;quad+=q
            valid=ids!=np.iinfo(np.int64).max
            scores=np.where(valid,scores,np.where(HIGH[None,:,None],-np.inf,np.inf))
            merged_ids=np.concatenate([best_ids,ids],axis=2)
            merged_scores=np.concatenate([best_scores,scores],axis=2)
            order=np.lexsort((merged_ids,np.where(HIGH[None,:,None],-merged_scores,merged_scores)),axis=2)[...,:K]
            best_ids=np.take_along_axis(merged_ids,order,axis=2)
            best_scores=np.take_along_axis(merged_scores,order,axis=2)
            done=records[index]["start"]+records[index]["n"]
            if index%20==0 or index+1==len(records):
                elapsed=time.monotonic()-began
                progress(out,stage="statistics",records=index+1,total_records=len(records),
                         tokens=done,total_tokens=total,neurons=n_units,elapsed_s=elapsed,
                         eta_s=elapsed*(total-done)/done)
    if not np.all(hist.sum(axis=2)==total) or not np.all(quad.sum(axis=(1,2))==total):
        raise ValueError("Population count mismatch")
    np.savez(out/"scan-state.npz",hist=hist,quadrants=quad,indices=best_ids,scores=best_scores)
    return hist,quad,best_ids,best_scores

def make_band_plan(hist,seed):
    n_units=len(hist);cuts=np.stack([core.quantiles(h[2],core.PERCENTILES) for h in hist])
    lookup=np.empty((n_units,65536),dtype=np.uint8)
    counts=np.zeros((n_units,14),dtype=np.uint64)
    ranks=[]
    for col in range(n_units):
        # Nonfinite bit patterns never occur in validated captures; mapping remains numeric.
        finite=np.flatnonzero((np.arange(65536)&0x7f80)!=0x7f80)
        lookup[col].fill(0)
        lookup[col,finite]=np.clip(np.searchsorted(cuts[col],core.floats(finite.astype(np.uint16)),side="right")-1,0,13)
        np.add.at(counts[col],lookup[col],hist[col,2])
        rng=np.random.default_rng(seed)
        ranks.append([np.sort(rng.choice(int(n),min(int(n),8),replace=False)) for n in counts[col]])
    return cuts,lookup,counts,ranks

def second_pass(records,n_units,hist,out,workers,seed):
    cuts,lookup,counts,ranks=make_band_plan(hist,seed)
    used=np.zeros_like(counts);positions=[[[] for _ in range(14)] for _ in range(n_units)]
    def bin_record(record):
        if DEVICES:return gpu_record(record,n_units,lookup)
        raw=read_record(record,n_units)
        bins=lookup[np.arange(n_units)[:,None],raw[:,:,2]]
        local_counts=np.stack([np.bincount(row,minlength=14) for row in bins]).astype(np.uint64)
        return bins,local_counts
    began=time.monotonic();total=sum(r["n"] for r in records)
    with pool_for(workers) as pool:
        for index,(bins,local_counts) in enumerate(ordered_map(pool,bin_record,records,workers)):
            for col in range(n_units):
                for band in range(14):
                    selected=ranks[col][band]
                    a=np.searchsorted(selected,used[col,band])
                    b=np.searchsorted(selected,int(used[col,band])+int(local_counts[col,band]))
                    if b>a:
                        local=np.flatnonzero(bins[col]==band)
                        chosen=local[selected[a:b]-int(used[col,band])]+records[index]["start"]
                        positions[col][band].extend(chosen.tolist())
            used+=local_counts
            if index%20==0 or index+1==len(records):
                elapsed=time.monotonic()-began;done=records[index]["start"]+records[index]["n"]
                progress(out,stage="quantile_samples",records=index+1,total_records=len(records),
                         tokens=done,total_tokens=total,neurons=n_units,elapsed_s=elapsed,
                         eta_s=elapsed*(total-done)/done)
    if not np.array_equal(used,counts):raise ValueError("Band denominator mismatch")
    result=[]
    for col in range(n_units):
        result.append([dict(low=core.PERCENTILES[i],high=core.PERCENTILES[i+1],
                            lower=float(cuts[col,i]),upper=float(cuts[col,i+1]),
                            count=int(counts[col,i]),indices=positions[col][i],upper_inclusive=i==13)
                       for i in range(14)])
    core.write(out/"band-state.json",result)
    return result

class CaptureView:
    """Read-only logical single-neuron view of original record arrays."""
    def __init__(self,records,column):
        self.records,self.column=records,column
        self.starts=np.array([r["start"] for r in records])
        self.total=sum(r["n"] for r in records)
    def __len__(self):return self.total
    @lru_cache(maxsize=16)
    def load(self,index):
        return np.load(self.records[index]["activation"],mmap_mode="r",allow_pickle=False)
    def __getitem__(self,key):
        scalar=isinstance(key,(int,np.integer))
        start,stop=(int(key),int(key)+1) if scalar else (key.start or 0,key.stop)
        if not 0<=start<stop<=self.total:raise IndexError("Capture view bounds")
        ri=int(np.searchsorted(self.starts,start,side="right"))-1
        record=self.records[ri]
        if stop>record["start"]+record["n"]:raise ValueError("Document crosses capture record")
        result=self.load(ri)[start-record["start"]:stop-record["start"],self.column,:]
        return result[0] if scalar else result

def render_report(data,out):
    data["_histogram_path"]=str(out/"histograms.npz")
    blocks=core.paper_blocks(data,PROFILE)
    b=blocks["bundle"];identity=data["identity"]
    b["claim"]["title"]=("SOFTWARE FIXTURE · " if all(doc["source"].get("source_file")=="software-fixture" for doc in data["documents"].values()) else "")+identity["id"]+" · "+str(data["n_tokens"])+" saved tokens"
    b["claim"]["lede"]=f'{data["n_tokens"]:,} tokens · {data["n_documents"]:,} documents · saved native gate/value/product. Interpretation: —.'
    b["neuron"]["model"]=identity["model"]
    full=b["exemplars"];rng=np.random.default_rng(PROFILE["display_seed"])
    shown={"top":full["top"][:10],"bottom":full["bottom"][:8],"bands":{}}
    for spec in PROFILE["bands"]:
        candidates=full["bands"][spec["source_key"]]
        # Fixed deterministic display subset; does not pretend to recover author sampling.
        chosen=sorted(rng.choice(len(candidates),min(5,len(candidates)),replace=False).tolist()) if candidates else []
        shown["bands"][spec["key"]]=[candidates[i] for i in chosen]
    b["exemplars"],b["all_exemplars"],b["display_profile"]=shown,full,PROFILE
    b["distribution"]["quantiles"]=[q for q in b["distribution"]["quantiles"] if q["quantile"] not in ("p0","p100")]
    b["local_metadata"]={
        "statistics":data["statistics"],"sign_counts":data["gate_value_sign_counts"],
        "scatter_uniform_n":data["scatter_uniform_n"],"scatter_total":len(data["scatter"]),
        "population_documents":data["n_documents"],
    }
    b["appendix"]={"notes":[],"provenance":[]}
    contexts={}
    for pid,p in data["points"].items():
        doc=data["documents"][p["document"]];j=p["position"]
        contexts[pid]={"position":j,"token":doc["tokens"][j] or "⟦token "+str(doc["token_ids"][j])+"⟧","values":p["values"],
                       "token_id":doc["token_ids"][j],"raw_token":doc["raw_tokens"][j],
                       "prefix_tail_ids":doc["token_ids"][max(0,j-4):j],"prefix_tail_raw":doc["raw_tokens"][max(0,j-4):j],
                       "prefix":"".join(doc["tokens"][:j]),"following":"".join(doc["tokens"][j+1:j+33]),
                       "source":doc["source"]}
    core.write(out/"paper-data.json",blocks)
    template=Path(__file__).with_name("report.html").read_text()
    for key,value in blocks.items():
        payload=json.dumps(value,ensure_ascii=False,allow_nan=False).replace("<","\\u003c").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
        marker="__PAPER_DATA_"+key+"__"
        if template.count(marker)!=1:raise ValueError("Template marker mismatch")
        template=template.replace(marker,payload)
    template=re.sub(r"<title>.*?</title>",lambda _: "<title>"+html.escape(b["claim"]["title"])+"</title>",template,count=1)
    payload=json.dumps(contexts,ensure_ascii=False,allow_nan=False).replace("<","\\u003c").replace("\u2028","\\u2028").replace("\u2029","\\u2029")
    adapter=Path(__file__).with_name("basic_adapter.js").read_text()
    template=template.replace("</body>",'<script id="local-contexts" type="application/json">'+payload+"</script><script>"+adapter+"</script></body>")
    if "__PAPER_DATA_" in template:raise ValueError("Unfilled template")
    (out/"index.html").write_text(template)
    del data["_histogram_path"]

def check_report(data,hist):
    n=data["n_tokens"]
    if any(s["n"]!=n for s in data["statistics"].values()) or sum(b["count"] for b in data["bands"])!=n:
        raise ValueError("Report denominators disagree")
    for band in data["bands"]:
        if len(band["points"])!=min(8,band["count"]):raise ValueError("Missing band samples")
        for key in band["points"]:
            x=data["points"][key]["values"][2]
            if not (band["lower"]<=x and (x<=band["upper"] if band["upper_inclusive"] else x<band["upper"])):
                raise ValueError("Out-of-band sample")
    if not np.all(hist.sum(axis=1)==n):raise ValueError("Histogram denominator")

def catalog(out,entries,complete):
    core.write(out/"catalog.json",{"complete":complete,"neurons":entries})
    ui_version=hashlib.sha256(Path(__file__).with_name("basic_adapter.js").read_bytes()).hexdigest()[:12]
    rows="".join('<tr><td><a href="'+html.escape(e["id"])+'/index.html?ui='+ui_version+'">'+html.escape(e["id"])+"</a></td><td>"+str(e["tokens"])+"</td><td>"+str(e["minimum"])+"</td><td>"+str(e["maximum"])+"</td></tr>" for e in entries)
    (out/"index.html").write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Saved neuron activations</title><style>body{max-width:1000px;margin:48px auto;padding:0 24px;font:16px/1.6 system-ui;color:#222}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:8px;border-bottom:1px solid #ddd}a{color:inherit}input{padding:8px;width:280px}</style><h1>Saved neuron activations</h1><p>'+str(len(entries))+' reports</p><input placeholder="Filter neuron ID" oninput="document.querySelectorAll(\'tbody tr\').forEach(r=>r.hidden=!r.textContent.toLowerCase().includes(this.value.toLowerCase()))"><table><thead><tr><th>Neuron</th><th>Tokens</th><th>Minimum</th><th>Maximum</th></tr></thead><tbody>'+rows+"</tbody></table></html>")

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("run","inputs","tokenizer","output"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--seed",type=int,default=42)
    parser.add_argument("--devices",help="Optional comma-separated physical GPU IDs for exact array counting; no model inference")
    args=parser.parse_args()
    global DEVICES
    if args.devices:
        physical=args.devices.split(",")
        if not all(re.fullmatch(r"[0-9]+",x) for x in physical) or len(set(physical))!=len(physical) or args.workers!=len(physical):
            parser.error("--devices requires distinct GPU IDs, one per worker")
        os.environ["CUDA_VISIBLE_DEVICES"]=args.devices
        import torch
        if torch.cuda.device_count()!=len(physical):parser.error("Requested CUDA devices unavailable")
        torch.set_num_threads(1)
        DEVICES=["cuda:"+str(i) for i in range(len(physical))]
    else:DEVICES=[]
    if not 1<=args.workers<=16:parser.error("workers must be 1..16")
    args.output.mkdir(parents=True,exist_ok=False)
    began=time.monotonic()
    try:
        target_doc=core.read(args.run/"targets.json");targets=target_doc["neurons"]
        if not targets or len({n["id"] for n in targets})!=len(targets):raise ValueError("Target identities must be unique")
        for target in targets:
            if target["id"]!=f'L{target["layer"]}_N{target["neuron"]}' or not re.fullmatch(r"L[0-9]+_N[0-9]+",target["id"]):
                raise ValueError("Expected zero-based dense Lx_Ny identity")
        records,signatures,_,_,n_docs=core.inventory(args.run,args.inputs,targets[0]["id"])
        signatures[str(args.tokenizer)]=core.signature(args.tokenizer)
        plan=core.read(args.run/"plan.json")
        core.write(args.output/"sources.json",{"records":records,"signatures":signatures,"targets":[{k:n[k] for k in ("id","layer","neuron")} for n in targets]})
        source_files=["build_reports.py","build_report.py","basic_adapter.js","report.html"]
        code_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_files}
        core.write(args.output/"manifest.json",{"tokens":sum(r["n"] for r in records),"documents":n_docs,"neurons":len(targets),"records":len(records),"seed":args.seed,"workers":args.workers,"counting_backend":"CUDA integer arrays" if DEVICES else "NumPy CPU","physical_gpu_ids":args.devices,"code_sha256":code_hashes,"new_model_forwards":0,"llm_calls":0})
        hist,quad,indices,scores=first_pass(records,len(targets),args.output,args.workers)
        bands=second_pass(records,len(targets),hist,args.output,args.workers,args.seed)
        entries=[]
        for col,target in enumerate(targets):
            out=args.output/target["id"];out.mkdir(exist_ok=False)
            np.savez(out/"histograms.npz",counts=hist[col],gate_value_sign_counts=quad[col])
            extremes={}
            for slot,key in enumerate(KEYS):
                extremes[key]=[{"index":int(i),"value":float(s)} for i,s in zip(indices[col,slot],scores[col,slot]) if i!=np.iinfo(np.int64).max]
            core.write(out/"extremes.json",extremes)
            identity={k:target[k] for k in ("id","layer","neuron")}
            identity["model"]=Path(plan["model"]).name
            view=CaptureView(records,col)
            data=core.package(records,view,hist[col],quad[col],extremes,bands[col],
                              args.inputs,args.tokenizer,identity,plan,args.seed,n_docs)
            check_report(data,hist[col])
            core.write(out/"report.json",data)
            render_report(data,out)
            entry={"id":identity["id"],"tokens":data["n_tokens"],"documents":n_docs,"minimum":data["statistics"]["product"]["min"],"maximum":data["statistics"]["product"]["max"],"points":len(data["points"]),"bands":len(data["bands"])}
            entries.append(entry);catalog(args.output,entries,False)
            core.write(out/"complete.json",{"complete":True,**entry,"new_model_forwards":0,"llm_calls":0})
            progress(args.output,stage="render",neurons_complete=len(entries),neurons=len(targets),elapsed_s=time.monotonic()-began)
            del data,view
        for path,sig in signatures.items():
            if core.signature(path)!=sig:raise ValueError("Source metadata changed: "+path)
        for name,expected in code_hashes.items():
            if hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()!=expected:
                raise ValueError("Report implementation changed during the run: "+name)
        catalog(args.output,entries,True)
        core.write(args.output/"complete.json",{"complete":True,"neurons":len(entries),"tokens_per_neuron":sum(r["n"] for r in records),"documents":n_docs,"source_metadata_unchanged":True,"full_source_content_rehash":False,"new_model_forwards":0,"llm_calls":0,"elapsed_s":time.monotonic()-began})
        progress(args.output,stage="complete",neurons_complete=len(entries),neurons=len(entries),elapsed_s=time.monotonic()-began)
    except Exception as exc:
        core.write(args.output/"failure.json",{"type":type(exc).__name__,"message":str(exc)})
        progress(args.output,stage="failed",error=type(exc).__name__)
        raise

if __name__=="__main__":main()
