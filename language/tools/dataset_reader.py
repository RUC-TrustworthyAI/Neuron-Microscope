"""Read native neuron activations from a saved corpus archive."""
from pathlib import Path
import json
import numpy as np
import pyarrow.parquet as pq

CHANNELS={"gate":0,"value":1,"product":2,"gate_raw":0,"value_raw":1,"native_product":2}

def open_dataset(directory):
    """Use a dataset-supplied adapter when present, otherwise the bundled reader."""
    adapter=Path(directory)/'reader.py'
    if adapter.is_file():
        import importlib.util
        spec=importlib.util.spec_from_file_location('saved_dataset_adapter',adapter)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        return module.Dataset(directory)
    return Dataset(directory)
def floats(bits):
    return (np.asarray(bits).astype(np.uint32)<<np.uint32(16)).view(np.float32)

class Dataset:
    def __init__(self,directory,*,_allow_incomplete=False):
        self.root=Path(directory)
        self.manifest=json.loads((self.root/"dataset.json").read_text())
        if not self.manifest["complete"] and not _allow_incomplete:
            raise ValueError("Dataset conversion is not complete")
        self.neurons=json.loads((self.root/self.manifest["neurons"]).read_text())
        self.columns={n["id"]:j for j,n in enumerate(self.neurons)}
        self.locations={n["id"]:(n["part"],n["part_column"]) for n in self.neurons}
        self.parts={p["id"]:p for p in self.manifest["parts"]}
        self.documents=pq.read_table(self.root/self.manifest["corpus"]["documents"])
        self.document_ids=self.documents["document_id"].to_numpy()
        self.starts=self.documents["token_start"].to_numpy()
        self.token_ids=np.load(self.root/self.manifest["corpus"]["token_ids"],mmap_mode="r",allow_pickle=False)

    def document(self,document_id):
        j=int(np.searchsorted(self.document_ids,document_id))
        if j==len(self.document_ids) or int(self.document_ids[j])!=document_id:
            raise KeyError(document_id)
        return self.documents.slice(j,1).to_pylist()[0]

    def _shard(self,part,shard_id):
        p=self.parts[part]
        s=p["shards"][shard_id]
        if s["id"]!=shard_id:raise ValueError("Invalid shard index")
        return self.root/p["path"]/s["path"]

    def document_activations(self,document_id,neuron_id):
        d=self.document(document_id)
        part,column=self.locations[neuron_id]
        a=np.load(self._shard(part,d["shard_id"]),mmap_mode="r",allow_pickle=False)
        try:
            start=d["shard_token_start"]
            values=floats(a[column,start:start+d["token_count"],:])
        finally:a._mmap.close()
        ids=np.array(self.token_ids[d["token_start"]:d["token_start"]+d["token_count"]],copy=True)
        return {"document":d,"token_ids":ids,"activations":values,"channels":self.manifest["channels"]}

    def iter_neuron(self,neuron_id):
        part,column=self.locations[neuron_id]
        for s in self.parts[part]["shards"]:
            a=np.load(self._shard(part,s["id"]),mmap_mode="r",allow_pickle=False)
            try:values=floats(a[column,:,:])
            finally:a._mmap.close()
            yield s["token_start"],values

    def extremes(self,neuron_id,channel="product",tail="high",k=5):
        if tail not in ("high","low"):raise ValueError("tail must be high or low")
        if not 0<=k<=self.manifest["indices"]["k"]:raise ValueError("Requested k exceeds the stored index")
        j=self.columns[neuron_id];c=CHANNELS[channel]
        with np.load(self.root/self.manifest["indices"]["extremes"],allow_pickle=False) as idx:
            locations=idx[tail+"_token_indices"][j,c,:k]
            values=idx[tail+"_values"][j,c,:k]
        result=[]
        for token,value in zip(locations,values):
            i=int(np.searchsorted(self.starts,int(token),side="right"))-1
            if i<0:raise ValueError("Invalid token index")
            d=self.documents.slice(i,1).to_pylist()[0]
            if not d["token_start"]<=int(token)<d["token_start"]+d["token_count"]:raise ValueError("Invalid token-to-document index")
            result.append({"dataset_token_index":int(token),"document_id":d["document_id"],
                "token_position":int(token)-d["token_start"],"value":float(value)})
        return result

    def histogram(self,neuron_id,channel="product"):
        part,column=self.locations[neuron_id];p=self.parts[part]
        a=np.load(self.root/p["path"]/p["indices"]["histogram"],mmap_mode="r",allow_pickle=False)
        try:return np.array(a[column,CHANNELS[channel]],copy=True)
        finally:a._mmap.close()

    def statistics(self,neuron_id):
        if neuron_id not in self.columns:raise KeyError(neuron_id)
        return pq.read_table(self.root/self.manifest["indices"]["statistics"],
            filters=[("neuron_id","=",neuron_id)]).to_pylist()

    def close(self):
        if self.token_ids is not None:
            self.token_ids._mmap.close();self.token_ids=None

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
