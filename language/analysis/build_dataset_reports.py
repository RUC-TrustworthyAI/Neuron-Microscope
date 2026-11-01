"""Build original-layout activation reports from a unified saved dataset; no model or LLM."""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from functools import lru_cache
import argparse,fcntl,hashlib,importlib.util,json,multiprocessing as mp,os,shutil,time,traceback
import numpy as np
import build_report as core
import build_reports as batch

API=None
INPUTS=None
TOKENIZER=None
PLAN=None
SEED=42
OUTPUT=None
TOKEN_TABLES=False
TOKEN_STATE=None

def load_reader(dataset):
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"tools"))
    from dataset_reader import open_dataset
    return open_dataset(dataset)

def initialize(dataset,inputs,tokenizer,output,seed,token_tables=False,token_cache=None):
    global API,INPUTS,TOKENIZER,OUTPUT,SEED,PLAN,TOKEN_TABLES,TOKEN_STATE
    API=load_reader(dataset);INPUTS=Path(inputs);TOKENIZER=Path(tokenizer);OUTPUT=Path(output);SEED=seed
    TOKEN_TABLES=token_tables
    if TOKEN_TABLES:
        from report_token_fields import TokenFields
        TOKEN_STATE=TokenFields(token_cache)
    d=API.manifest
    PLAN=dict(d["capture"],model=d["model"],model_snapshot=d["model_snapshot"],
        dataset_seed=d["selection"]["original_dataset_seed"],input_run=str(INPUTS.parent),
        max_tokens=d["capture"]["max_document_tokens"])

class NeuronView(batch.CaptureView):
    def __getitem__(self,key):
        scalar=isinstance(key,(int,np.integer))
        start,stop=(int(key),int(key)+1) if scalar else (key.start or 0,key.stop)
        if not 0<=start<stop<=self.total:raise IndexError("Dataset view bounds")
        ri=int(np.searchsorted(self.starts,start,side="right"))-1;r=self.records[ri]
        if stop>r["start"]+r["n"]:raise ValueError("Document crosses a shard")
        value=self.load(ri)[self.column,start-r["start"]:stop-r["start"],:]
        return value[0] if scalar else value

def records_for(unit):
    part=API.parts[unit["part"]]
    return [dict(block=s["id"],start=s["token_start"],n=s["token_count"],
        activation=str(API.root/part["path"]/s["path"]),
        offsets=str(INPUTS/("block-%06d.offsets.u64.npy"%s["id"])),rows=None) for s in part["shards"]]

def scan_neuron(unit):
    hist=np.stack([API.histogram(unit["id"],name) for name in core.CHANNELS])
    n=API.manifest["tokens"]
    assert hist.dtype==np.uint64 and np.all(hist.sum(axis=1)==n)
    cuts,lookup,counts,ranks=batch.make_band_plan(hist[None,...],SEED)
    cuts,lookup,counts,ranks=cuts[0],lookup[0],counts[0],ranks[0]
    used=np.zeros(14,np.uint64);positions=[[] for _ in range(14)]
    quadrant=np.zeros((3,3),np.uint64)
    best_ids=np.full((4,batch.K),np.iinfo(np.int64).max,np.int64)
    best_scores=np.broadcast_to(np.where(batch.HIGH[:,None],-np.inf,np.inf),(4,batch.K)).copy()
    records=records_for(unit);seen=0
    token_fields=TOKEN_STATE.start(hist[2]) if TOKEN_TABLES else None
    for rec,(start,values) in zip(records,API.iter_neuron(unit["id"]),strict=True):
        assert start==seen==rec["start"] and values.shape==(rec["n"],3) and np.isfinite(values).all()
        if token_fields is not None:token_fields.update(API.token_ids[start:start+len(values)],values[:,2])
        g,v=values[:,0],values[:,1]
        gs=np.where(g<0,0,np.where(g>0,2,1));vs=np.where(v<0,0,np.where(v>0,2,1))
        quadrant+=np.bincount(gs*3+vs,minlength=9).reshape(3,3).astype(np.uint64)
        ids,scores=batch.extreme_products(values[:,2][None,:],rec)
        ids,scores=ids[0],scores[0]
        valid=ids!=np.iinfo(np.int64).max
        scores=np.where(valid,scores,np.where(batch.HIGH[:,None],-np.inf,np.inf))
        merged_ids=np.concatenate([best_ids,ids],axis=1);merged_scores=np.concatenate([best_scores,scores],axis=1)
        order=np.lexsort((merged_ids,np.where(batch.HIGH[:,None],-merged_scores,merged_scores)),axis=1)[:,:batch.K]
        best_ids=np.take_along_axis(merged_ids,order,axis=1);best_scores=np.take_along_axis(merged_scores,order,axis=1)
        product_bits=(values[:,2].copy().view(np.uint32)>>16).astype(np.uint16)
        bins=lookup[product_bits];local_counts=np.bincount(bins,minlength=14).astype(np.uint64)
        for band in range(14):
            a=np.searchsorted(ranks[band],used[band])
            b=np.searchsorted(ranks[band],int(used[band])+int(local_counts[band]))
            if b>a:
                local=np.flatnonzero(bins==band)
                chosen=local[ranks[band][a:b]-int(used[band])]+start
                positions[band].extend(chosen.tolist())
        used+=local_counts;seen+=len(values)
        if rec["block"]%128==0 or seen==n:
            core.write(OUTPUT/"jobs"/(unit["id"]+".json"),dict(stage="statistics",tokens=seen,total_tokens=n,shards=rec["block"]+1))
    assert seen==n and np.array_equal(used,counts) and quadrant.sum()==n
    for channel,marginal in ((0,quadrant.sum(axis=1)),(1,quadrant.sum(axis=0))):
        wanted=np.bincount(batch.SIGN,minlength=3,weights=hist[channel]).astype(np.uint64)
        assert np.array_equal(marginal,wanted)
    assert best_scores[0,0]==core.statistics(hist[2])["max"] and best_scores[1,0]==core.statistics(hist[2])["min"]
    bands=[dict(low=core.PERCENTILES[i],high=core.PERCENTILES[i+1],
        lower=float(cuts[i]),upper=float(cuts[i+1]),count=int(counts[i]),
        indices=positions[i],upper_inclusive=i==13) for i in range(14)]
    extremes={key:[dict(index=int(i),value=float(s)) for i,s in zip(best_ids[slot],best_scores[slot])
        if i!=np.iinfo(np.int64).max] for slot,key in enumerate(batch.KEYS)}
    return hist,quadrant,extremes,bands,records,token_fields

def dataset_index(folder,data,api):
    """Preserve opaque report case IDs while mapping every context to the shared dataset."""
    documents={}
    for key,doc in data["documents"].items():
        docid=int(doc["source"]["document_order"])
        assert str(docid)==str(key)
        actual=api.document_activations(docid,data["identity"]["id"])
        assert np.array_equal(np.asarray(doc["token_ids"],np.uint32),actual["token_ids"])
        wanted=np.column_stack([doc[name] for name in ("gate","value","product")]).astype(np.float32)
        assert np.array_equal(wanted.view(np.uint32),actual["activations"].view(np.uint32))
        documents[key]=dict(document_id=docid,token_start=actual["document"]["token_start"],token_count=len(wanted))
    points={}
    for key,p in data["points"].items():
        doc=data["documents"][p["document"]];d=documents[p["document"]];at=int(p["position"])
        assert 0<=at<d["token_count"]
        values=np.array([doc[name][at] for name in ("gate","value","product")],np.float32)
        assert np.array_equal(values.view(np.uint32),np.array(p["values"],np.float32).view(np.uint32))
        points[key]=dict(document_id=d["document_id"],token_position=at,dataset_token_index=d["token_start"]+at)
    payload=dict(schema_version=1,neuron=data["identity"]["id"],source_dataset=str(api.root),
        case_id_semantics="Report-local opaque identifiers; use document_id and token_position for shared lookup",
        documents=documents,points=points,all_saved_context_token_ids_and_channel_bits_equal=True)
    core.write(folder/"dataset-index.json",payload)
    return "dataset-index.json"

def source_signatures(api,inputs,tokenizer):
    paths={api.root/"dataset.json",api.root/"neurons.json",Path(tokenizer)}
    if (api.root/"reader.py").exists():paths.add(api.root/"reader.py")
    for key in ("documents","token_ids"):
        paths.add(api.root/api.manifest["corpus"][key])
    for part in api.parts.values():
        paths.add(api.root/part["path"]/part["indices"]["histogram"])
        for shard in part["shards"]:
            paths.add(api.root/part["path"]/shard["path"])
            for suffix in ("ids.u32.npy","offsets.u64.npy","documents.json"):
                paths.add(Path(inputs)/("block-%06d.%s"%(shard["id"],suffix)))
    return {str(p):core.signature(p) for p in sorted(paths)}

def publish_catalog(out,entries,complete):
    batch.catalog(out,entries,complete)
    path=out/"catalog.json";data=core.read(path)
    data["case_id_semantics"]="Report-local opaque identifiers"
    data["shared_dataset_mapping"]="<neuron-id>/dataset-index.json"
    core.write(path,data)

def build_one(unit):
    began=time.monotonic();out=OUTPUT/unit["id"]
    assert not out.exists(),"Refuse to overwrite an existing report"
    out.mkdir()
    hist,quad,extremes,bands,records,token_fields=scan_neuron(unit)
    np.savez(out/"histograms.npz",counts=hist,gate_value_sign_counts=quad)
    core.write(out/"extremes.json",extremes)
    core.write(out/"band-anchors.json",bands)
    view=NeuronView(records,unit["part_column"])
    identity={k:unit[k] for k in ("id","layer","neuron")};identity["model"]=API.manifest.get("model_name","Qwen3-32B")
    data=core.package(records,view,hist,quad,extremes,bands,INPUTS,TOKENIZER,identity,PLAN,SEED,API.manifest["documents"])
    batch.check_report(data,hist)
    for key,point in data["points"].items():
        expected=core.floats(view[int(key)])
        assert np.array_equal(expected,np.array(point["values"],np.float32))
    core.write(out/"report.json",data);batch.render_report(data,out)
    if token_fields is not None:token_fields.publish(out,unit["id"])
    mapping=dataset_index(out,data,API)
    assert not __import__("re").search(r"__PAPER_DATA_[A-Za-z0-9_-]+__", (out/"index.html").read_text())
    entry=dict(id=unit["id"],tokens=data["n_tokens"],documents=data["n_documents"],
        minimum=data["statistics"]["product"]["min"],maximum=data["statistics"]["product"]["max"],
        points=len(data["points"]),bands=len(data["bands"]),dataset_index=mapping)
    core.write(out/"complete.json",dict(complete=True,**entry,new_model_forwards=0,llm_calls=0,
        source_dataset=str(API.root),wall_s=time.monotonic()-began))
    view.load.cache_clear()
    core.write(OUTPUT/"jobs"/(unit["id"]+".json"),dict(stage="complete",tokens=API.manifest["tokens"],wall_s=time.monotonic()-began))
    return entry

def copy_existing(source,out):
    if source is None:return [],[]
    done=core.read(source/"complete.json");assert done["complete"]
    entries=core.read(source/"catalog.json")["neurons"];assert len(entries)==done["neurons"]
    copied=[]
    for e in entries:
        src=source/e["id"];dst=out/e["id"];assert not dst.exists()
        assert __import__("re").fullmatch(r"L[0-9]+_N[0-9]+",e["id"])
        assert core.read(src/"complete.json")["complete"] and (src/"index.html").is_file()
        signatures={str(p.relative_to(src)):core.signature(p) for p in src.rglob("*") if p.is_file()}
        assert not any(p.is_symlink() for p in src.rglob("*"))
        shutil.copytree(src,dst)
        for name,signature in signatures.items():
            original=src/name;new=dst/name
            assert core.signature(original)==signature
            assert hashlib.sha256(original.read_bytes()).digest()==hashlib.sha256(new.read_bytes()).digest()
        copied.append(dict(id=e["id"],files=len(signatures),all_file_bytes_equal=True))
        batch.catalog(out,entries[:len(copied)],False)
    return entries,copied

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ("dataset","inputs","tokenizer","output"):parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--existing",type=Path);parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--token-tables",action="store_true");parser.add_argument("--token-cache",type=Path)
    parser.add_argument("--seed",type=int,default=42);parser.add_argument("--wait",action="store_true")
    args=parser.parse_args();assert 1<=args.workers<=32
    if args.token_tables:assert args.token_cache and args.token_cache.is_dir()
    expected=int(core.read(args.dataset/"dataset.json")["neuron_count"])
    args.output.mkdir(parents=True,exist_ok=False);(args.output/"jobs").mkdir()
    lock=(args.output/".build.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    began=time.monotonic()
    sources=("build_dataset_reports.py","build_report.py","build_reports.py","basic_adapter.js","report.html","report_token_fields.py","add_token_tables.py","token_tables.js","token_tables.css")
    hashes={n:hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest() for n in sources}
    core.write(args.output/"manifest.json",dict(complete=False,expected_neurons=expected,
        dataset=str(args.dataset),existing=str(args.existing),code_sha256=hashes,
        workers=args.workers,seed=args.seed,new_model_forwards=0,llm_calls=0))
    try:
        batch.progress(args.output,stage="copy_existing",neurons_complete=0,neurons=expected)
        entries,copied=copy_existing(args.existing,args.output)
        core.write(args.output/"existing-copy-verification.json",dict(complete=True,reports=copied))
        batch.progress(args.output,stage="waiting_for_complete_dataset",neurons_complete=len(entries),neurons=expected)
        deadline=time.time()+24*3600
        while not (args.dataset/"verification.json").exists() or not core.read(args.dataset/"dataset.json")["complete"]:
            if not args.wait:raise RuntimeError("Dataset must complete verification before reporting")
            if time.time()>deadline:raise TimeoutError("Dataset wait deadline")
            time.sleep(10)
        verified=core.read(args.dataset/"verification.json");assert verified["complete"]
        data=load_reader(args.dataset);units=data.neurons;manifest=data.manifest
        signatures=source_signatures(data,args.inputs,args.tokenizer)
        core.write(args.output/"source-signatures.json",signatures)
        assert len(units)==len({u["id"] for u in units})==manifest["neuron_count"]==expected
        existing={e["id"] for e in entries};assert len(existing)==len(entries) and existing<={u["id"] for u in units}
        assert all(e["tokens"]==manifest["tokens"] and e["documents"]==manifest["documents"] for e in entries)
        for entry in entries:
            folder=args.output/entry["id"]
            entry["dataset_index"]=dataset_index(folder,core.read(folder/"report.json"),data)
        pending=[u for u in units if u["id"] not in existing]
        data.close()
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context("spawn"),initializer=initialize,
                initargs=(str(args.dataset),str(args.inputs),str(args.tokenizer),str(args.output),args.seed,args.token_tables,str(args.token_cache))) as pool:
            futures={pool.submit(build_one,u):u["id"] for u in pending}
            try:
                for future in as_completed(futures):
                    entries.append(future.result());entries.sort(key=lambda e:tuple(int(x) for x in __import__("re").findall(r"\d+",e["id"])))
                    publish_catalog(args.output,entries,False)
                    batch.progress(args.output,stage="render",neurons_complete=len(entries),neurons=len(units),
                        new_reports_complete=len(entries)-len(existing),new_reports=len(pending),elapsed_s=time.monotonic()-began)
            except BaseException:
                for future in futures:future.cancel()
                pool.shutdown(wait=True,cancel_futures=True)
                raise
        assert len(entries)==expected and {e["id"] for e in entries}=={u["id"] for u in units}
        for path,signature in signatures.items():assert core.signature(path)==signature,"Saved source changed: "+path
        for name,digest in hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==digest
        for e in entries:
            folder=args.output/e["id"];done=core.read(folder/"complete.json")
            assert done["complete"] and done["tokens"]==manifest["tokens"] and (folder/"index.html").is_file()
        publish_catalog(args.output,entries,True)
        metadata=core.read(args.output/"manifest.json");metadata.update(complete=True,tokens=manifest["tokens"],documents=manifest["documents"],neurons=len(units))
        core.write(args.output/"manifest.json",metadata)
        core.write(args.output/"complete.json",dict(complete=True,neurons=expected,copied_existing=len(existing),
            generated_new=len(pending),tokens_per_neuron=manifest["tokens"],documents=manifest["documents"],
            source_dataset=str(args.dataset),source_metadata_unchanged=True,source_files_checked=len(signatures),full_source_content_rehash=False,new_model_forwards=0,llm_calls=0,elapsed_s=time.monotonic()-began))
        batch.progress(args.output,stage="complete",neurons_complete=expected,neurons=expected)
    except BaseException:
        core.write(args.output/"failure.json",dict(complete=False,error=traceback.format_exc()))
        batch.progress(args.output,stage="failed")
        raise

if __name__=="__main__":main()
