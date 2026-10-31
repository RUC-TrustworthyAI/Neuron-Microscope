import numpy as np
import torch
from vision_repro.all_neuron_examples import LAYERS, extrema, selected_ranks, quantize_heat


def test_all_layers():
    assert len(LAYERS)==12
    assert sum(c for _,c,_ in LAYERS)==5804


def test_exact_signed_extrema_and_dead_relu():
    pre=torch.tensor([[[[-3.,-1.],[-2.,-4.]],[[2.,5.],[-8.,1.]]]])
    values,locations=extrema(pre.relu(),pre)
    assert values.tolist()==[[[0.,-4.],[5.,-8.]]]
    assert locations.tolist()==[[[0,3],[1,2]]]


def test_rank_windows_and_ties():
    scores=np.zeros((600,2,2),np.float32)
    scores[:,1,0]=np.arange(600)
    scores[:,1,1]=-np.arange(600)
    indices=selected_ranks(scores)
    assert indices.shape==(2,48)
    assert indices[0,:16].tolist()==list(range(16))
    assert indices[0,16:24].tolist()==list(range(46,54))
    assert indices[0,24:32].tolist()==list(range(496,504))
    assert indices[1,:16].tolist()==list(range(599,583,-1))
    assert indices[1,32:].tolist()==list(range(599,583,-1))


def test_heat_quantization_signed_and_zero():
    raw=np.array([[-4.,0.],[2.,4.]],np.float32)
    q,p=quantize_heat(raw);assert p==4 and q.tolist()==[[0,0],[128,255]]
    q,p=quantize_heat(raw,True);assert p==4 and q.tolist()==[[255,0],[0,0]]
    q,p=quantize_heat(np.zeros((2,2)));assert p==0 and not q.any()
