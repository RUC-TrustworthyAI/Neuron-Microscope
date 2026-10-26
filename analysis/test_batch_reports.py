"""Saved-capture fixture tests; never load a model or call an LLM."""
from pathlib import Path
import json,tempfile,unittest,os
from unittest.mock import patch
import numpy as np
import build_reports as batch
import build_report as core

ROOT=Path(__file__).resolve().parents[1]
TOKENIZER=None

def fixture(root,n_docs=16):
    global TOKENIZER
    n_tokens=n_docs*64;half=n_tokens//2
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders
    TOKENIZER=root/"tokenizer.json"
    tokenizer=Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder=decoders.ByteLevel()
    tokenizer.train_from_iterator([" hello"*64], trainers.BpeTrainer(vocab_size=300, special_tokens=["[UNK]"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tokenizer.save(str(TOKENIZER))
    run=root/"run";inputs=root/"inputs";run.mkdir();inputs.mkdir()
    targets=[{"id":f"L0_N{i}","layer":0,"neuron":i,"evidence":"FORBIDDEN_SEMANTIC_LABEL"} for i in range(4)]
    core.write(run/"targets.json",{"neurons":targets})
    plan={"target_global_tokens":n_tokens,"model":"/fixture/Qwen3-32B","model_snapshot":"software-fixture",
          "input_run":str(inputs),"dataset_seed":42,"max_tokens":64,"batch_size":1,"attention":"fixture","dtype":"bfloat16"}
    core.write(run/"plan.json",plan)
    token=Tokenizer.from_file(str(TOKENIZER)).encode(" hello",add_special_tokens=False).ids[0]
    np.save(inputs/"block-000000.ids.u32.npy",np.full(n_tokens,token,dtype=np.uint32))
    np.save(inputs/"block-000000.offsets.u64.npy",np.arange(0,n_tokens+1,64,dtype=np.uint64))
    core.write(inputs/"block-000000.documents.json",[{"document_order":i,"source_file":"software-fixture","url":"https://example.invalid/doc/"+str(i),"input_row":i,"tokens":64} for i in range(n_docs)])
    paths=[];raws=[]
    for replica in range(2):
        folder=run/f"replica-{replica}";folder.mkdir()
        prefix=folder/"block-000000"
        rows=np.arange(replica,n_docs,2,dtype=np.uint32)
        off=np.arange(0,half+1,64,dtype=np.uint64)
        x=np.arange(half,dtype=np.float32)+replica*half
        values=np.zeros((half,4,3),dtype=np.float32)
        values[:,0,0]=(x%9)-4;values[:,0,1]=(x%13)-6;values[:,0,2]=(x%17)-8
        values[:,2,:]=2
        values[:,3,0]=-0.;values[:,3,1]=-3;values[:,3,2]=np.where(x%7==0,-0.,-.5)
        bits=(values.view(np.uint32)>>16).astype(np.uint16)
        activation=Path(str(prefix)+".activations.bf16.u16.npy")
        np.save(activation,bits);raws.append(bits)
        np.save(str(prefix)+".document_rows.u32.npy",rows);np.save(str(prefix)+".offsets.u64.npy",off)
        path=Path(str(prefix)+"-complete.json")
        core.write(path,{"block":0,"documents":n_docs//2,"tokens":half,"shape":[half,4,3],"activation_file":str(activation)})
        paths.append(str(path))
    core.write(run/"checkpoint.json",{"ranks":[{"complete_paths":paths}]})
    records,signatures,_,_,n_docs=core.inventory(run,inputs,"L0_N0")
    return run,inputs,targets,records,signatures,n_docs,np.concatenate(raws),plan

class BatchTests(unittest.TestCase):
    def setUp(self):
        (ROOT/".local").mkdir(exist_ok=True)
        physical=os.environ.get("BASIC_REPORT_TEST_DEVICES")
        if physical:
            os.environ["CUDA_VISIBLE_DEVICES"]=physical
            import torch
            torch.set_num_threads(1)
            batch.DEVICES=["cuda:"+str(i) for i in range(len(physical.split(",")))]
            self.workers=len(batch.DEVICES)
            batch.THREAD.device=batch.DEVICES[0]
        else:
            batch.DEVICES=[]
            self.workers=2

    def test_full_capture_math_sampling_and_native_contexts(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".local",prefix="basic-report-fixture-") as tmp:
            root=Path(tmp);run,inputs,targets,records,signatures,n_docs,raw,plan=fixture(root)
            output=root/"out";output.mkdir()
            hist,quad,indices,scores=batch.first_pass(records,4,output,self.workers)
            bands=batch.second_pass(records,4,hist,output,self.workers,42)
            for col in range(4):
                for channel in range(3):
                    np.testing.assert_array_equal(hist[col,channel],np.bincount(raw[:,col,channel],minlength=65536))
                decoded=core.floats(raw[:,col,:])
                expected_quad=np.zeros((3,3),np.uint64)
                for gate,value in decoded[:,:2]:expected_quad[int(np.sign(gate))+1,int(np.sign(value))+1]+=1
                np.testing.assert_array_equal(quad[col],expected_quad)
                for high,slot in ((True,0),(False,1)):
                    expected=sorted(range(1024),key=lambda i:((-decoded[i,2] if high else decoded[i,2]),i))[:30]
                    self.assertEqual(indices[col,slot].tolist(),expected)
                    doc_candidates=[]
                    for r in records:
                        for start in range(0,r["n"],64):
                            values=decoded[r["start"]+start:r["start"]+start+64,2]
                            pos=r["start"]+start+int((np.argmax if high else np.argmin)(values))
                            doc_candidates.append(pos)
                    expected=sorted(doc_candidates,key=lambda i:((-decoded[i,2] if high else decoded[i,2]),i))
                    got=indices[col,slot+2];got=got[got!=np.iinfo(np.int64).max]
                    self.assertEqual(got.tolist(),expected)
                expected_bands=core.band_samples(raw[:,col,:],hist[col],42)
                self.assertEqual(bands[col],expected_bands)
                out=output/targets[col]["id"];out.mkdir()
                np.savez(out/"histograms.npz",counts=hist[col],gate_value_sign_counts=quad[col])
                extremes={key:[{"index":int(i),"value":float(s)} for i,s in zip(indices[col,k],scores[col,k]) if i!=np.iinfo(np.int64).max] for k,key in enumerate(batch.KEYS)}
                identity={k:targets[col][k] for k in ("id","layer","neuron")};identity["model"]="Qwen3-32B"
                view=batch.CaptureView(records,col)
                data=core.package(records,view,hist[col],quad[col],extremes,bands[col],inputs,TOKENIZER,identity,plan,42,n_docs)
                batch.check_report(data,hist[col]);batch.render_report(data,out)
                for index,p in data["points"].items():
                    self.assertEqual(p["values"],decoded[int(index)].tolist())
                    doc=data["documents"][p["document"]]
                    r=records[int(p["document"])%2]
                    expected_doc_start=r["start"]+(int(p["document"])//2)*64
                    np.testing.assert_array_equal(np.array([doc[c] for c in core.CHANNELS]).T,decoded[expected_doc_start:expected_doc_start+64])
                page=(out/"index.html").read_text()
                blocks=core.read(out/"paper-data.json")
                b=blocks["bundle"]
                self.assertNotIn("FORBIDDEN_SEMANTIC_LABEL",page)
                if data["statistics"]["product"]["min"]==data["statistics"]["product"]["max"]:
                    self.assertIn("data-constant-frequency",page)
                    self.assertEqual(b["bigscan"]["hist"],[len(raw)])
                self.assertIn("SOFTWARE FIXTURE",b["claim"]["title"])
                self.assertEqual(b["claim"]["one_paragraph"],"—")
                descriptions=[note for note in b["appendix"]["notes"] if note.startswith("p") and "uniformly sampled anchors." in note]
                self.assertEqual(len(descriptions),14)
                for band in data["bands"]:
                    key="p%s–%s: " % (band["low"],band["high"])
                    matching=[note for note in descriptions if note.startswith(key)]
                    self.assertEqual(len(matching),1)
                    self.assertIn(str(band["count"])+" tokens;",matching[0])
                self.assertIsNone(b["tests"]);self.assertIsNone(b["circuit"])
                self.assertLessEqual(len(b["exemplars"]["top"]),10)
                self.assertLessEqual(len(b["exemplars"]["bottom"]),8)
                self.assertEqual(len(b["exemplars"]["bands"]),7)
                self.assertTrue(all(len(v)<=5 for v in b["exemplars"]["bands"].values()))
                for channel in core.CHANNELS:
                    stats=data["statistics"][channel]
                    self.assertEqual(len(stats["histogram"]["edges"]),len(stats["histogram"]["counts"])+1)
            self.assertTrue(all(core.signature(p)==s for p,s in signatures.items()))

    def test_small_capture(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".local",prefix="basic-report-fixture-") as tmp:
            root=Path(tmp);run,inputs,targets,records,_,n_docs,raw,plan=fixture(root,2)
            output=root/"out";output.mkdir()
            hist,quad,indices,scores=batch.first_pass(records,4,output,self.workers)
            bands=batch.second_pass(records,4,hist,output,self.workers,42)
            for col in range(4):
                out=output/targets[col]["id"];out.mkdir()
                np.savez(out/"histograms.npz",counts=hist[col],gate_value_sign_counts=quad[col])
                extremes={key:[{"index":int(i),"value":float(s)} for i,s in zip(indices[col,k],scores[col,k])
                               if i!=np.iinfo(np.int64).max] for k,key in enumerate(batch.KEYS)}
                identity={**targets[col],"model":"Qwen3-32B"}
                data=core.package(records,batch.CaptureView(records,col),hist[col],quad[col],extremes,bands[col],
                                  inputs,TOKENIZER,identity,plan,42,n_docs)
                batch.check_report(data,hist[col]);batch.render_report(data,out)
                self.assertEqual(data["scatter_uniform_n"],128)
                self.assertEqual(len(set(data["scatter"])),128)
                self.assertEqual(len(data["examples"]["top_documents"]),2)

    def test_duplicate_document_preflight_fails(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".local",prefix="basic-report-fixture-") as tmp:
            root=Path(tmp);run,inputs,_,records,*_=fixture(root)
            p=Path(records[-1]["rows"]);rows=np.load(p);rows[-1]=rows[-2];np.save(p,rows)
            output=root/"out"
            argv=["build_reports.py","--run",str(run),"--inputs",str(inputs),
                  "--tokenizer",str(TOKENIZER),"--output",str(output)]
            with patch("sys.argv",argv),self.assertRaisesRegex(ValueError,"Duplicate document row"):
                batch.main()
            self.assertEqual(core.read(output/"progress.json")["stage"],"failed")
            self.assertEqual(core.read(output/"failure.json")["type"],"ValueError")
            self.assertFalse((output/"complete.json").exists())

    def test_unicode_fragments_keep_native_token_identity(self):
        from tokenizers import Tokenizer
        with tempfile.TemporaryDirectory(dir=ROOT/".local",prefix="basic-report-fixture-") as tmp:
            root=Path(tmp);run,inputs,targets,records,_,n_docs,raw,plan=fixture(root,2)
            tokenizer=Tokenizer.from_file(str(TOKENIZER))
            emoji=tokenizer.encode("\U0001fabf",add_special_tokens=False).ids
            self.assertGreater(len(emoji),1)
            ids=np.resize(np.array(emoji,dtype=np.uint32),128)
            np.save(inputs/"block-000000.ids.u32.npy",ids)
            output=root/"out";output.mkdir()
            hist,quad,indices,scores=batch.first_pass(records,4,output,self.workers)
            bands=batch.second_pass(records,4,hist,output,self.workers,42)
            out=output/"L0_N0";out.mkdir()
            np.savez(out/"histograms.npz",counts=hist[0],gate_value_sign_counts=quad[0])
            extremes={key:[{"index":int(i),"value":float(s)} for i,s in zip(indices[0,k],scores[0,k])
                           if i!=np.iinfo(np.int64).max] for k,key in enumerate(batch.KEYS)}
            data=core.package(records,batch.CaptureView(records,0),hist[0],quad[0],extremes,bands[0],
                              inputs,TOKENIZER,{"id":"L0_N0","layer":0,"neuron":0,"model":"Qwen3-32B"},plan,42,n_docs)
            self.assertTrue(any(not t for d in data["documents"].values() for t in d["tokens"]))
            for doc in data["documents"].values():
                self.assertEqual("".join(doc["tokens"]),tokenizer.decode(doc["token_ids"],skip_special_tokens=False))
            batch.render_report(data,out)
            b=core.read(out/"paper-data.json")["bundle"]
            items=b["all_exemplars"]["top"]+b["all_exemplars"]["bottom"]+sum(b["all_exemplars"]["bands"].values(),[])
            for item in items:
                self.assertTrue(item["peak_token"])
                doc=data["documents"][data["points"][item["id"]]["document"]]
                for i,token in enumerate(item["token_activations"],item["window"]["start"]):
                    self.assertTrue(token["token"])
                    self.assertEqual(token["token_id"],doc["token_ids"][i])
                    self.assertEqual(token["raw_token"],doc["raw_tokens"][i])

    def test_nonfinite_and_cross_record_reads_fail(self):
        with tempfile.TemporaryDirectory(dir=ROOT/".local",prefix="basic-report-fixture-") as tmp:
            root=Path(tmp);run,inputs,targets,records,_,_,_,_=fixture(root)
            view=batch.CaptureView(records,0)
            with self.assertRaises(ValueError):view[500:520]
            p=Path(records[0]["activation"]);array=np.load(p);array[0,0,2]=0x7fc0;np.save(p,array)
            with self.assertRaises(ValueError):batch.scan_record(records[0],4)

if __name__=="__main__":unittest.main()
