import torch
from vision_repro.five_variants import gaussian_weights,channel_scores,path_vector,unit_mean_anchor,diversity_penalty
from vision_repro.targets import NeuronTarget
from vision_repro.five_variants import appearance_regularizers
from vision_repro.five_variants import reweight_fft_scale
from vision_repro.rgb_explore import logit_color_gain


def test_appearance_zero_defaults_and_constant_color():
    x=torch.rand(1,3,16,16,requires_grad=True)
    assert appearance_regularizers(x,{})[0]==0
    c=torch.ones_like(x)*.7
    loss,contrast,chroma=appearance_regularizers(c,{'luma_contrast_weight':1.,'chroma_detail_weight':1.})
    assert contrast==0 and chroma<1e-12 and loss<1e-12


def test_appearance_has_finite_gradients_without_luma_chroma_leakage():
    gray=torch.rand(1,1,16,16).expand(-1,3,-1,-1).clone().requires_grad_(True)
    loss,contrast,chroma=appearance_regularizers(gray,{'luma_std_limit':.1,'luma_contrast_weight':1.,'chroma_detail_weight':1.})
    assert contrast>0 and chroma<1e-12
    loss.backward();assert torch.isfinite(gray.grad).all() and gray.grad.abs().sum()>0


def test_fft_power_preserves_coefficient_grid_energy():
    scale=torch.tensor([1.,2.,4.,8.])
    assert reweight_fft_scale(scale,1) is scale
    adjusted=reweight_fft_scale(scale,1.5)
    assert torch.allclose(adjusted.square().mean(),scale.square().mean())
    assert torch.allclose(adjusted[-1]/adjusted[0],torch.tensor(8.**1.5))


def test_logit_color_gain_preserves_gray_and_has_gradients():
    x=torch.randn(1,1,8,8).expand(-1,3,-1,-1).clone()
    assert torch.allclose(logit_color_gain(x,2.),x,atol=1e-6)
    z=torch.randn(1,3,8,8,requires_grad=True)
    assert logit_color_gain(z,1.) is z
    mapped=logit_color_gain(z,1.5);mapped.sigmoid().mean().backward()
    assert torch.isfinite(z.grad).all() and z.grad.abs().sum()>0


def test_batched_gaussian_matches_existing_definition():
    x=torch.randn(4,5,14,14,requires_grad=True)
    assert torch.allclose(channel_scores(x,2,.12).mean(),NeuronTarget('mixed4c',2,'gaussian',.12)(x))
    assert torch.allclose(gaussian_weights(x).sum((-2,-1)),torch.ones(4))


def test_negative_minimizes_signed_pre_not_zero_post():
    pre=torch.full((1,3,5,5),-3.,requires_grad=True)
    loss=channel_scores(pre,1).mean();loss.backward()
    assert (pre.grad[:,1]>0).all()
    assert channel_scores(pre.relu(),1).item()==0
    assert channel_scores(pre-.1*pre.grad,1)<loss


def test_path_zeros_target_and_is_unit_scale_invariant():
    a=torch.rand(2,5,7,7,requires_grad=True);b=torch.rand(2,6,7,7,requires_grad=True)
    x=path_vector(a,b,2);c=b.detach().clone();c[:,2]=100000
    assert torch.allclose(x,path_vector(a,c,2),atol=1e-6)
    assert torch.allclose(x,path_vector(a*99,b*.1,2),atol=1e-6)
    assert torch.allclose(x.norm(dim=1),torch.ones(2))
    x[:,0].sum().backward()
    assert b.grad[:,2].count_nonzero()==0
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0


def test_anchor_equal_weights_unit_paths_not_raw_strength():
    x=torch.tensor([[100.,0.],[0.,1.]])
    assert torch.allclose(unit_mean_anchor(x),torch.ones(2)/2**.5)


def test_diversity_identical_vs_orthogonal_grams():
    same=torch.ones(4,3,5,5,requires_grad=True)
    assert torch.allclose(diversity_penalty(same),torch.tensor(1.))
    orth=torch.zeros(4,4,5,5)
    for i in range(4):orth[i,i]=1
    assert diversity_penalty(orth)==0
    x=torch.rand(4,6,5,5,requires_grad=True);diversity_penalty(x).backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    assert diversity_penalty(x[:1])==0


def test_paths_at_peak_have_correct_spatial_weight():
    x=torch.ones(2,3,7,7);centers=torch.tensor([[-.5,-.5],[.5,.5]])
    w=gaussian_weights(x,.12,centers)
    assert w[0].argmax()==0 and w[1].argmax()==48
    assert torch.isfinite(path_vector(torch.zeros_like(x),torch.zeros_like(x),1)).all()
