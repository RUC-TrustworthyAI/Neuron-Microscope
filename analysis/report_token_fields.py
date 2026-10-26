"""Token selectivity from the same saved-activation pass used for report sampling."""
from pathlib import Path
import json
import numpy as np
import add_token_tables as tables

class TokenFields:
    def __init__(self,cache):
        self.cache=Path(cache)
        self.counts=np.load(self.cache/"token-counts.i64.npy",mmap_mode="r")
        self.labels=json.loads((self.cache/"token-labels.json").read_text())
    def start(self,hist):
        return TokenScan(self,hist)

class TokenScan:
    def __init__(self,parent,hist):
        self.parent=parent;self.hist=hist
        observed=tables.LEVELS[hist>0];self.maximum=float(observed.max());self.minimum=float(observed.min())
        self.pt=tables.FRACTIONS*max(0.,self.maximum);self.nt=tables.FRACTIONS*min(0.,self.minimum)
        vocab=len(parent.counts);self.sums=np.zeros(vocab,np.float64);self.squares=np.zeros(vocab,np.float64)
        self.positive=np.zeros((3,vocab),np.int64);self.negative=np.zeros_like(self.positive);self.seen=0
    def update(self,ids,product):
        ids=np.asarray(ids,np.int64)
        # Reuse the exact threshold and moment definitions of existing reports.
        bits=(np.asarray(product).view(np.uint32)>>16).astype(np.uint16)
        tables.update_counts(ids,bits,self.parent.counts,self.sums,self.squares,self.positive,self.negative,self.pt,self.nt)
        self.seen+=len(ids)
    def publish(self,out,neuron):
        counts=self.parent.counts;n=int(self.hist.sum())
        assert self.seen==n==int(counts.sum())
        checks=[]
        for k,f in enumerate(tables.FRACTIONS):
            ep=int(self.hist[tables.LEVELS>self.pt[k]].sum());en=int(self.hist[tables.LEVELS<self.nt[k]].sum())
            assert int(self.positive[k].sum())==ep and int(self.negative[k].sum())==en
            assert np.all(self.positive[k]<=counts) and np.all(self.negative[k]<=counts)
            checks.append(dict(fraction=float(f),positive_count=ep,negative_count=en))
        levels=np.where(np.isfinite(tables.LEVELS),tables.LEVELS,0).astype(np.float64)
        assert np.isclose(self.sums.sum(),np.dot(self.hist,levels),rtol=1e-9,atol=1e-4)
        assert np.isclose(self.squares.sum(),np.dot(self.hist,levels**2),rtol=1e-9,atol=1e-4)
        summary=dict(neuron=neuron,tokens=n,observed_token_ids=int(np.count_nonzero(counts)),positive_max=self.maximum,
            negative_min=self.minimum,fractions=tables.FRACTIONS.tolist(),positive_thresholds=self.pt.tolist(),
            negative_thresholds=self.nt.tolist(),minimum_count=50,ranking_size=80,histogram_checks=checks,
            comparisons={"positive":">","negative":"<"})
        ranked=tables.rankings(counts,self.sums,self.squares,self.positive,self.negative,self.pt,self.nt,self.parent.labels)
        np.savez(out/"token_stats.npz",cnt=counts,summ=self.sums,sumsq=self.squares,fractions=tables.FRACTIONS,
            positive_thresholds=self.pt,negative_thresholds=self.nt,positive_counts=self.positive,negative_counts=self.negative)
        tables.save(out/"token-rankings.json",ranked);tables.save(out/"token-summary.json",summary)
        a=Path(__file__).parent
        text=tables.insert_tables((out/"index.html").read_text(),ranked,summary,
            (a/"token_tables.css").read_text(),(a/"token_tables.js").read_text())
        (out/"index.html").write_text(text)
        tables.save(out/"token-tables-complete.json",dict(complete=True,neuron=neuron,tokens=n,tables=6,rows_per_table=80,minimum_count=50))
