"""Build a self-contained neuron report from an existing dense BF16 capture."""
from pathlib import Path
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import argparse, codecs, hashlib, json, time
import numpy as np

CHANNELS = ["gate", "value", "product"]
PERCENTILES = [0, .01, .1, 1, 5, 10, 25, 50, 75, 90, 95, 99, 99.9, 99.99, 100]

def read(p):
    return json.loads(Path(p).read_text())

def write(p, data):
    Path(p).write_text(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")

def floats(bits):
    return (np.asarray(bits, dtype=np.uint32) << 16).view(np.float32)

def signature(p):
    s = Path(p).stat()
    return [s.st_size, s.st_mtime_ns, s.st_ino, s.st_dev]

def quantiles(hist, ps):
    ids = np.flatnonzero(hist)
    values = floats(ids.astype(np.uint16)).astype(np.float64)
    order = np.argsort(values, kind="stable")
    values, counts = values[order], hist[ids[order]]
    cdf = np.cumsum(counts, dtype=np.uint64)
    ranks = np.asarray(ps, dtype=np.float64) / 100 * (int(cdf[-1]) - 1)
    lo, hi = np.floor(ranks).astype(np.int64), np.ceil(ranks).astype(np.int64)
    a = values[np.searchsorted(cdf, lo, side="right")]
    b = values[np.searchsorted(cdf, hi, side="right")]
    return a + (b - a) * (ranks - lo)

def statistics(hist):
    ids = np.flatnonzero(hist)
    x, n = floats(ids.astype(np.uint16)).astype(np.float64), hist[ids]
    total = int(n.sum())
    mean = float(np.dot(x, n) / total)
    variance = float(np.dot((x - mean) ** 2, n) / total)
    edges = np.unique(np.r_[x.min(), quantiles(hist, np.linspace(0, 100, 101)), x.max()])
    counts = np.histogram(x, bins=edges, weights=n)[0] if len(edges) > 1 else np.array([total])
    return dict(n=total, mean=mean, std=variance ** .5, min=float(x.min()), max=float(x.max()),
                positive=int(n[x > 0].sum()), negative=int(n[x < 0].sum()), zero=int(n[x == 0].sum()),
                quantiles=dict(zip(map(str, PERCENTILES), map(float, quantiles(hist, PERCENTILES)))),
                histogram=dict(edges=(edges.tolist() if len(edges)>1 else [float(edges[0]),float(edges[0])+max(abs(float(edges[0]))*.01,1.0)]), counts=counts.astype(np.uint64).tolist()))

def best(values, k, high):
    """Stable value ranking, then earliest source position; partition without sorting every token."""
    n = min(k, len(values))
    if not n:
        return np.array([], dtype=np.int64)
    z = -values if high else values
    threshold = np.partition(z, n - 1)[n - 1]
    better = np.flatnonzero(z < threshold)
    equal = np.flatnonzero(z == threshold)[:n - len(better)]
    ids = np.r_[better, equal]
    return ids[np.lexsort((ids, z[ids]))]

def trim(items, k, high):
    return sorted(items, key=lambda x: ((-x["value"] if high else x["value"]), x["index"]))[:k]

def inventory(run, inputs, neuron):
    targets = read(run / "targets.json")
    column = next(i for i, n in enumerate(targets["neurons"]) if n["id"] == neuron)
    cp = read(run / "checkpoint.json")
    paths = [Path(p) for rank in cp["ranks"] for p in rank["complete_paths"]]
    for host in ("124", "178"):
        paths += sorted((run / ("host-" + host)).glob("replica-*/block-*-complete.json"))
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate committed record")
    records, seen, start = [], set(), 0
    files = {str(p): signature(p) for p in [run / "targets.json", run / "checkpoint.json", run / "plan.json"]}
    for path in sorted(paths, key=lambda p: (int(p.name.split("-")[1]), str(p))):
        d = read(path)
        prefix = path.parent / path.name.replace("-complete.json", "")
        rows_path, off_path = Path(str(prefix) + ".document_rows.u32.npy"), Path(str(prefix) + ".offsets.u64.npy")
        rows, offsets = np.load(rows_path, allow_pickle=False), np.load(off_path, allow_pickle=False)
        canon = inputs / ("block-%06d" % d["block"])
        canonical_offsets = np.load(str(canon) + ".offsets.u64.npy", allow_pickle=False)
        lengths = np.diff(offsets)
        if rows.dtype != np.uint32 or offsets.dtype != np.uint64 or offsets[0] != 0 or np.any(lengths <= 0):
            raise ValueError("Invalid document map: " + str(path))
        if len(rows) != d["documents"] or not np.array_equal(lengths, np.diff(canonical_offsets)[rows]):
            raise ValueError("Document length mismatch")
        if len(np.unique(rows)) != len(rows):
            raise ValueError("Duplicate document row: " + str(path))
        orders = d["block"] * 1024 + rows.astype(np.int64)
        if any(int(x) in seen for x in orders):
            raise ValueError("Repeated document")
        seen.update(map(int, orders))
        activation = Path(d["activation_file"])
        n = int(offsets[-1])
        if n != d["tokens"] or d["shape"] != [n, len(targets["neurons"]), 3]:
            raise ValueError("Invalid activation dimensions")
        records.append(dict(record=str(path), activation=str(activation), block=d["block"],
                            rows=str(rows_path), offsets=str(off_path), start=start, n=n))
        start += n
        for p in [path, activation, rows_path, off_path]:
            files[str(p)] = signature(p)
    expected = read(run / "plan.json")["target_global_tokens"]
    if start != expected or seen != set(range(len(seen))):
        raise ValueError("Incomplete token or document coverage")
    for block in sorted({r["block"] for r in records}):
        for suffix in [".ids.u32.npy", ".offsets.u64.npy", ".documents.json"]:
            p = inputs / ("block-%06d%s" % (block, suffix))
            files[str(p)] = signature(p)
    return records, files, column, targets["neurons"][column], len(seen)

def collect(records, column, out, workers):
    total = sum(r["n"] for r in records)
    column_file = out / "column.bf16.u16.npy"
    cache = np.lib.format.open_memmap(column_file, mode="w+", dtype=np.uint16, shape=(total, 3))
    hist = np.zeros((3, 65536), dtype=np.uint64)
    extremes = {key: [] for key in ["top_tokens", "bottom_tokens", "top_documents", "bottom_documents"]}
    quadrants = np.zeros((3, 3), dtype=np.uint64)
    began = time.monotonic()
    def load(r):
        a = np.load(r["activation"], mmap_mode="r", allow_pickle=False)
        if a.dtype != np.uint16 or a.shape != (r["n"], 200, 3):
            raise ValueError("Invalid BF16 array")
        return np.array(a[:, column, :], copy=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for base in range(0, len(records), workers):
            group = records[base:base + workers]
            for r, bits in zip(group, pool.map(load, group)):
                if np.any((bits & 0x7f80) == 0x7f80):
                    raise ValueError("Non-finite captured value")
                cache[r["start"]:r["start"] + r["n"]] = bits
                for c in range(3):
                    hist[c] += np.bincount(bits[:, c], minlength=65536).astype(np.uint64)
                x = floats(bits)
                quadrants += np.bincount(((np.sign(x[:, 0]).astype(int) + 1) * 3 +
                                          np.sign(x[:, 1]).astype(int) + 1), minlength=9).reshape(3, 3).astype(np.uint64)
                product = x[:, 2]
                off = np.load(r["offsets"], allow_pickle=False)
                for high, prefix in [(True, "top"), (False, "bottom")]:
                    ids = best(product, 30, high)
                    key = prefix + "_tokens"
                    extremes[key] = trim(extremes[key] + [
                        dict(index=r["start"] + int(i), value=float(product[i])) for i in ids], 30, high)
                    scores = (np.maximum if high else np.minimum).reduceat(product, off[:-1].astype(np.int64))
                    key = prefix + "_documents"
                    candidates = []
                    for j in best(scores, 30, high):
                        a, b = map(int, off[j:j+2])
                        pos = a + int(np.flatnonzero(product[a:b] == scores[j])[0])
                        candidates.append(dict(index=r["start"] + pos, value=float(scores[j])))
                    extremes[key] = trim(extremes[key] + candidates, 30, high)
                done = r["start"] + r["n"]
                progress = dict(stage="scan", tokens=done, total=total, records=base+group.index(r)+1,
                                elapsed_s=time.monotonic()-began)
                progress["eta_s"] = progress["elapsed_s"] * (total-done) / done
                write(out / "progress.json", progress)
                if (base // workers) % 50 == 0:
                    print(json.dumps(progress), flush=True)
    cache.flush()
    np.savez(out / "histograms.npz", counts=hist, gate_value_sign_counts=quadrants)
    write(out / "extremes.json", extremes)
    return hist, quadrants, extremes, cache

def band_samples(cache, hist, seed, per_band=8):
    cuts = quantiles(hist[2], PERCENTILES)
    value_ids = np.flatnonzero(hist[2])
    assignments = np.clip(np.searchsorted(cuts, floats(value_ids.astype(np.uint16)), side="right")-1, 0, len(cuts)-2)
    counts = np.zeros(len(cuts)-1, dtype=np.uint64)
    np.add.at(counts, assignments, hist[2, value_ids])
    rng = np.random.default_rng(seed)
    ranks = [np.sort(rng.choice(int(n), min(int(n), per_band), replace=False)) for n in counts]
    positions, used = [[] for _ in counts], np.zeros(len(counts), dtype=np.uint64)
    for start in range(0, len(cache), 1_000_000):
        x = floats(cache[start:start+1_000_000, 2])
        bins = np.clip(np.searchsorted(cuts, x, side="right")-1, 0, len(cuts)-2)
        for band in range(len(counts)):
            local = np.flatnonzero(bins == band)
            a = np.searchsorted(ranks[band], used[band])
            b = np.searchsorted(ranks[band], int(used[band])+len(local))
            positions[band] += (start + local[ranks[band][a:b]-int(used[band])]).tolist()
            used[band] += len(local)
    if not np.array_equal(used, counts):
        raise ValueError("Band sample denominator mismatch")
    return [dict(low=PERCENTILES[i], high=PERCENTILES[i+1], lower=float(cuts[i]), upper=float(cuts[i+1]),
                 count=int(counts[i]), indices=positions[i], upper_inclusive=i==len(counts)-1)
            for i in range(len(counts))]

def package(records, cache, hist, quadrants, extremes, bands, inputs, tokenizer_path, identity, plan, seed, n_docs):
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
    # ByteLevel BPE fragments are streamed through one UTF-8 decoder, preserving split multibyte characters.
    byte_chars = list(range(33,127)) + list(range(161,173)) + list(range(174,256))
    extra = [x for x in range(256) if x not in byte_chars]
    inverse = {chr(c): c for c in byte_chars}
    inverse.update({chr(256+i): b for i,b in enumerate(extra)})
    def tokens_text(ids):
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        pieces, raw = [], []
        for token in ids:
            s = tokenizer.id_to_token(int(token))
            raw.append(s)
            if all(c in inverse for c in s):
                pieces.append(decoder.decode(bytes(inverse[c] for c in s), final=False))
            else:
                pieces.append(decoder.decode(b"", final=True) + s)
                decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        if pieces:
            pieces[-1] += decoder.decode(b"", final=True)
        if "".join(pieces) != tokenizer.decode(list(map(int,ids)), skip_special_tokens=False):
            raise ValueError("Tokenizer text reconstruction mismatch")
        return pieces, raw
    @lru_cache(maxsize=8)
    def block(b):
        prefix = inputs / ("block-%06d" % b)
        return (np.load(str(prefix)+".ids.u32.npy", mmap_mode="r", allow_pickle=False),
                np.load(str(prefix)+".offsets.u64.npy", allow_pickle=False),
                read(str(prefix)+".documents.json"))
    starts = np.array([r["start"] for r in records])
    documents, points = {}, {}
    def point(index):
        index = int(index)
        key = str(index)
        if key in points:
            return key
        r = records[int(np.searchsorted(starts, index, side="right"))-1]
        off = np.load(r["offsets"], allow_pickle=False)
        rows = np.load(r["rows"], allow_pickle=False) if r.get("rows") else np.arange(len(off)-1, dtype=np.uint32)
        local = index-r["start"]
        j = int(np.searchsorted(off,local,side="right"))-1
        row = int(rows[j]); ids, io, meta = block(r["block"])
        d = meta[row]; doc_id = str(d["document_order"])
        if doc_id not in documents:
            a,b = map(int,io[row:row+2])
            tokens = ids[a:b]
            texts, raw = tokens_text(tokens)
            act_start, act_stop = int(r["start"]+off[j]),int(r["start"]+off[j+1])
            values = floats(cache[act_start:act_stop])
            if len(tokens) != len(values):
                raise ValueError("Context/activation alignment mismatch")
            documents[doc_id] = dict(source=d, token_ids=tokens.tolist(), tokens=texts, raw_tokens=raw,
                                     gate=values[:,0].tolist(), value=values[:,1].tolist(), product=values[:,2].tolist())
        pos = int(local-off[j])
        points[key] = dict(document=doc_id, position=pos, values=floats(cache[index]).tolist())
        return key
    examples = {k:[point(x["index"]) for x in v] for k,v in extremes.items()}
    for b in bands:
        b["points"] = [point(i) for i in b.pop("indices")]
    scatter = [point(i) for i in np.sort(np.random.default_rng(seed+1).choice(len(cache), min(256,len(cache)), replace=False))]
    for key in list(points):
        if key not in scatter:
            scatter.append(key)
    return dict(identity=identity, channels=CHANNELS, n_tokens=len(cache), n_documents=n_docs,
                plan=plan, seed=seed, statistics={c:statistics(hist[i]) for i,c in enumerate(CHANNELS)},
                gate_value_sign_counts=quadrants.tolist(), examples=examples, bands=bands,
                scatter=scatter, scatter_uniform_n=min(256,len(cache)), points=points, documents=documents,
                methods=dict(quantiles="Exact empirical linear quantiles from native BF16 value counts.",
                             bands="Numeric intervals [lower, upper), final interval includes maximum. Duplicate cutoffs can create empty bands; ties are never split.",
                             sampling="Uniform token sampling without replacement inside each numeric band; seed fixed. Multiple anchors may come from one document.",
                             extremes="Exact token extrema and distinct-document extrema; ties ordered by scan index.",
                             scatter="Up to 256 uniform tokens, augmented by displayed extreme and band anchors; this is not a population-density plot.",
                             anatomy="Raw gate before SiLU, raw up/value, and captured native product; product is not recomputed.",
                             contexts="Actual captured input IDs, with per-document context reset and 4096-token truncation; no new forwards."))

def paper_blocks(d, profile=None):
    """Map our measurements into the original paper schema; absent evidence stays absent."""
    import html
    missing = "—"
    def window(point_id, kind="extra", pole=None):
        p = d["points"][str(point_id)]
        doc = d["documents"][p["document"]]
        j = p["position"]
        cfg = (profile or {}).get("windows", {}).get(kind)
        if cfg:
            b = min(len(doc["tokens"]), j+1+cfg["after"])
            a = max(0, b-cfg["max_tokens"], j-cfg.get("before",cfg["max_tokens"]-1))
        else:
            a, b = max(0,j-32), min(len(doc["tokens"]),j+33)
        display_pos = None
        if cfg and kind=="band" and profile.get("band_window_extreme"):
            values = doc["product"][a:b]
            display_pos = a+int(np.argmin(values) if pole=="negative" else np.argmax(values))
            # Removing only a suffix preserves the chosen extremum. The sampling anchor may leave
            # the preview; its actual input and readout remain separately accessible.
            b = min(b, display_pos+1+cfg["after"])
        # Expand only the prefix after selecting the original highlighted peak.
        if kind in ("top", "bottom", "band", "additional_band"):
            prefix_tokens = (profile or {}).get("context_prefix_tokens")
            if prefix_tokens is not None:
                peak = display_pos if display_pos is not None else j
                a = max(0, peak - prefix_tokens)
            else:
                a = max(0, j - (j-a) * (profile or {}).get("context_prefix_multiplier", 1))
        display_token = lambda i: doc["tokens"][i] or "⟦token "+str(doc["token_ids"][i])+"⟧"
        result = dict(id=str(point_id), text="".join(doc["tokens"][a:b]), activation=p["values"][2],
                    peak_index=j-a, peak_token=display_token(j),
                    token_activations=[dict(token=display_token(i),activation=doc["product"][i],token_id=doc["token_ids"][i],raw_token=doc["raw_tokens"][i]) for i in range(a,b)],
                    cut_left=a>0,cut_right=b<len(doc["tokens"]),source="corpus",
                    document_order=doc["source"]["document_order"], input_position=j,
                    gate=p["values"][0],up=p["values"][1],cluster=missing,group=missing,
                    tokens=[display_token(i) for i in range(a,b)],act=doc["product"][a:b],
                    gate_trace=doc["gate"][a:b],up_trace=doc["value"][a:b])
        result["window"] = dict(start=a, stop=b, selection_index=j-a, selection_visible=a<=j<b, policy="extra" if kind=="additional_band" else kind)
        if cfg and kind=="band" and profile.get("band_window_extreme"):
            pos = display_pos
            k = pos-a
            result["selection_anchor"] = dict(id=str(point_id),position=j,
                gate=p["values"][0],value=p["values"][1],product=p["values"][2],token=display_token(j),token_id=doc["token_ids"][j],raw_token=doc["raw_tokens"][j])
            result["display_peak"] = dict(id=str(int(point_id)+pos-j),position=pos,
                gate=doc["gate"][pos],value=doc["value"][pos],product=doc["product"][pos],
                token=display_token(pos),token_id=doc["token_ids"][pos],raw_token=doc["raw_tokens"][pos],operation="minimum" if pole=="negative" else "maximum",
                prefix="".join(doc["tokens"][:pos]))
            result.update(activation=doc["product"][pos],peak_index=k,peak_token=display_token(pos),
                          input_position=pos,gate=doc["gate"][pos],up=doc["value"][pos])
        return result
    top = [window(p,"top") for p in d["examples"]["top_documents"]]
    bottom = [window(p,"bottom") for p in d["examples"]["bottom_documents"]]
    bands = {}
    descriptions = []
    for band in reversed(d["bands"]):
        key = "p%s–%s" % (band["low"],band["high"])
        main_source = any(x["source_key"]==key for x in (profile or {}).get("bands", []))
        bands[key] = [window(p, "band" if main_source else "additional_band",
                             "negative" if band["upper"]<=0 else "positive") for p in band["points"]]
        descriptions.append("%s: [%s, %s%s; %s tokens; %s uniformly sampled anchors." %
                            (key,band["lower"],band["upper"],"]" if band["upper_inclusive"] else ")",band["count"],len(band["points"])))
    points = [window(p) for p in d["scatter"]]
    anatomy_points = [dict(p, text=html.escape(p["text"]),peak_token=html.escape(p["peak_token"])) for p in points]
    # Tooltip strings in the original renderer are HTML; escape corpus text at this boundary.
    q = {key:0 for key in ["g+u+","g+u-","g-u+","g-u-"]}
    for p in points:
        key = ("g+" if p["gate"]>0 else "g-")+("u+" if p["up"]>0 else "u-")
        q[key] += 1
    strip = top[0]
    s = d["statistics"]["product"]
    qs = [dict(quantile="p"+str(p),value=s["quantiles"][str(p)]) for p in PERCENTILES]
    # Reuse the original renderer's full-population histogram panel.
    counts = np.load(d["_histogram_path"],allow_pickle=False)["counts"][2]
    ids = np.flatnonzero(counts)
    values = floats(ids.astype(np.uint16))
    centers,inverse = np.unique(values,return_inverse=True)
    hh = np.zeros(len(centers),dtype=np.uint64)
    np.add.at(hh,inverse,counts[ids])
    name = d["identity"]["id"]
    corpus_size = "1B" if d["n_tokens"]==1_000_000_000 else format(d["n_tokens"],",")+" tokens"
    fixture = all(doc["source"].get("source_file")=="software-fixture" for doc in d["documents"].values())
    title = ("SOFTWARE FIXTURE · " if fixture else "")+name+" · FineWeb "+corpus_size
    b = dict(neuron=dict(model=d["identity"]["model"],layer=d["identity"]["layer"],neuron=d["identity"]["neuron"],id=d["identity"]["id"],
                         autointerp_label=missing,autointerp_description=missing),
             version=dict(n=None,ts=None,history=[]),
             claim=dict(title=title,lede="%s tokens · %s document sequences · BF16, batch 1, SDPA auto. Missing data: —." %
                        (format(d["n_tokens"],","),format(d["n_documents"],",")),
                        one_paragraph=missing,negative_pole=missing,what_would_change_my_mind=missing,confidence=missing),
             distribution=dict(mean=s["mean"],std=s["std"],n_tokens=d["n_tokens"],corpus_min=s["min"],corpus_max=s["max"],quantiles=qs),
             exemplars=dict(top=top,bottom=bottom,bands=bands),
             search=dict(corpus_max=s["max"],corpus_min=s["min"],regions=[]),
             anatomy=dict(points=anatomy_points,n=len(points),quadrants={missing:q},
                          strip=dict(text=strip["text"],tokens=strip["tokens"],gate=strip["gate_trace"],up=strip["up_trace"],act=strip["act"]),
                          headline=[]),
             headline=dict(strip=dict(text=strip["text"],tokens=strip["tokens"],values=strip["act"],
                                      caption="A saved natural-corpus context at the strongest document peak; no synthetic probe.")),
             bigscan=dict(n_tokens=d["n_tokens"],n_docs=d["n_documents"],corpus="FineWeb sample-10BT",
                          hist=hh.tolist(),hist_centers=centers.tolist(),min=s["min"],max=s["max"],
                          harvest_min=None,harvest_max=None,top=[],bottom=[],quantiles={}),
             spectrum=None,routes=[],routes_matrix=None,routes_meta=None,clusters=[],family=None,
             circuit=None,logit_effect=None,ablation=None,graphs_curated=None,custom=None,tests=None,criticism=[],
             appendix=dict(notes=["— denotes missing evidence or an experiment not performed. No semantic labels or agent explanations were generated.",
                                   "Extreme lists show 30 distinct documents per pole, ranked by the saved native product. Exact token extrema are retained in extremes.json.",
                                   "Anatomy: 256 uniform token anchors plus the union of token extrema, document extrema and band anchors (six token extrema are not among the 172 audited cases). Cluster labels are —; the quadrant table counts these displayed points, not the whole corpus. Its g-/u- columns include exact zero, matching the original renderer.",
                                   "Full-population gate/value sign counts (negative, zero, positive) are retained in report.json and histograms.npz.",
                                   "Both distribution panels use the same saved 1B-token population; the original author's separate harvest reference is —.",
                                   "Quantiles: exact empirical linear quantiles from native BF16 counts. Numeric bands keep tied BF16 values together; empty bands remain empty. Anchor, not context maximum, determines band membership.",
                                   *descriptions],
                           provenance=["Source capture: "+d["plan"]["input_run"],"Model snapshot: "+d["plan"]["model_snapshot"],
                                       "Uniform document sampling seed %s; per-document cap %s tokens." % (d["plan"]["dataset_seed"],d["plan"]["max_tokens"]),
                                       "Raw gate before SiLU, raw up/value, captured native product. No product recomputation or model forwards.",
                                       "Template: Goodfire neuron paper; all original scientific payloads replaced."]))
    review = dict(group_name=missing,organized=missing,understood=missing,confidence=None,one_line=missing,
                  main_job=missing,where_it_branches=[dict(branch=missing,grows_from=missing,path=missing,size=missing,evidence=missing)],
                  remainders=[dict(what=missing,share=missing,evidence=missing)],concerning=[missing],fine=[missing],
                  poles={p:dict(group=missing,strong_range=missing,summary=missing,double_negatives=missing) for p in ["positive","negative"]})
    return {"bundle":b,"paper-review-data":review,"paper-review-full-data":review}

def render(out):
    # One activation-only renderer for both single-neuron and dataset builders.
    from build_reports import render_report
    render_report(read(out/"report.json"), out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--neuron")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if args.render_only:
        render(args.output)
        return
    if not all([args.run,args.inputs,args.tokenizer,args.neuron]) or not 1 <= args.workers <= 8:
        parser.error("--run, --inputs, --tokenizer, --neuron and 1–8 workers required")
    args.output.mkdir(parents=True, exist_ok=False)
    began = time.monotonic()
    records, signatures, column, identity, n_docs = inventory(args.run,args.inputs,args.neuron)
    write(args.output/"sources.json", dict(records=records, signatures=signatures, column=column))
    hist, quadrants, extremes, cache = collect(records,column,args.output,args.workers)
    write(args.output/"progress.json",dict(stage="sampling",tokens=len(cache),total=len(cache)))
    bands = band_samples(cache,hist,args.seed)
    data = package(records,cache,hist,quadrants,extremes,bands,args.inputs,args.tokenizer,identity,
                   read(args.run/"plan.json"),args.seed,n_docs)
    for p,s in signatures.items():
        if signature(p) != s:
            raise ValueError("Source changed during analysis: "+p)
    write(args.output/"report.json",data)
    render(args.output)
    del cache
    (args.output/"column.bf16.u16.npy").unlink()
    write(args.output/"progress.json",dict(stage="complete",tokens=data["n_tokens"],total=data["n_tokens"],
                                         documents=n_docs, elapsed_s=time.monotonic()-began,
                                         original_files_unchanged=True,new_model_forwards=0))
    print((args.output/"progress.json").read_text(),flush=True)

if __name__ == "__main__":
    main()
