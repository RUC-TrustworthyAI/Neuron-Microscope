"""Full-image FFT exploration. No alpha, display compositing, or sample guidance."""
import csv
import hashlib
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .models import load_inceptionv1, preprocess
from .positive import ScheduledFFT, gaussian_blur
from .optimize import augment, save_image
from .targets import NeuronTarget
from .relu_redirect import RedirectState, install_redirected_relu
from .color_statistics import load_fitted_color

DEFAULTS=dict(size=124,steps=768,seed=0,color_mode='lucid_svd',std=.01,lr=.03,
    gaussian_sigma=.2,uniform_mix=0.,final_band=1.,initial_band=.15,unlock_until=.7,
    blur_start=2.,blur_end=.5,logit_l2=0.,mean_color_weight=0.,mean_color=[.68,.75,.78],
    mapping='sigmoid',mapping_gain=1.,jitter=4,rotation=5.,scale_min=.95,scale_max=1.05,
    activation_cap=0.,view_size=0,periphery_weight=0.,periphery_sigma=.35,
    periphery_color=[.68,.76,.79],tv_weight=0.,periphery_mean_weight=0.,
    periphery_tv_weight=0.,periphery_power=1.,layer='mixed4c',channel=447,
    common_view_size=0,schedule_steps=None,save_steps=(64,192,384,768))


def optimization_schedule(step, config):
    """Keep the original annealing horizon when testing additional updates.

    A null schedule_steps retains the historical full-run schedule. An explicit
    horizon finishes annealing there and holds final values for later updates.
    """
    horizon=config['steps'] if config['schedule_steps'] is None else config['schedule_steps']
    if horizon<1 or step<0:
        raise ValueError('Schedule horizon must be positive and step nonnegative')
    progress=min(step/max(horizon-1,1),1.)
    fraction=min(progress/config['unlock_until'],1)
    cutoff=config['initial_band']+(config['final_band']-config['initial_band'])*(1-math.cos(math.pi*fraction))/2
    sigma=config['blur_start']+(config['blur_end']-config['blur_start'])*min(progress/.85,1)
    lr=config['lr']*(.1+.9*(1+math.cos(math.pi*min(max((progress-.5)/.5,0),1)))/2)
    return cutoff,sigma,lr


def spatial_score(features, config):
    g=NeuronTarget(config['layer'],config['channel'],'gaussian',config['gaussian_sigma'])(features)
    mean=NeuronTarget(config['layer'],config['channel'],'mean')(features)
    return (1-config['uniform_mix'])*g+config['uniform_mix']*mean


def activation_reward(score, config):
    """Optional capped reward: stop paying for activation above a stated threshold.

    This is our regularized feasibility experiment, not a recovered author setting.
    Reported activation remains uncapped and uses the same evaluation definition.
    """
    cap=config['activation_cap']
    if cap<0:
        raise ValueError('Activation cap must be nonnegative')
    return score.clamp(max=cap) if cap else score


def model_view(image, size):
    """A stated observation scale, not a larger optimized or displayed canvas."""
    if not size or image.shape[-2:]==(size,size):
        return image
    return F.interpolate(image,size=(size,size),mode='bilinear',align_corners=False,antialias=True)


def periphery_mask(image, config):
    sigma=config['periphery_sigma']
    if sigma<=0:
        raise ValueError('Periphery sigma must be positive')
    h,w=image.shape[-2:]
    y=torch.linspace(-.5,.5,h,device=image.device,dtype=image.dtype)
    x=torch.linspace(-.5,.5,w,device=image.device,dtype=image.dtype)
    r2=y[:,None].square()+x[None,:].square()
    weight=(1-torch.exp(-r2/(2*sigma**2))).pow(config['periphery_power'])
    return weight[None,None]


def periphery_penalty(image, config):
    """Smooth radial RGB prior; center has zero weight, no alpha/compositing."""
    weight=periphery_mask(image,config)
    color=image.new_tensor(config['periphery_color'])[None,:,None,None]
    return ((image-color).square()*weight).mean()


def periphery_statistics(image, config):
    """Constrain regional color means, or regional variation, without compositing."""
    weight=periphery_mask(image,config)
    avg=(image*weight).sum((2,3))/weight.sum((2,3)).clamp_min(1e-12)
    color=(avg-image.new_tensor(config['periphery_color'])).square().mean()
    dx=image[:,:,:,1:]-image[:,:,:,:-1]
    dy=image[:,:,1:,:]-image[:,:,:-1,:]
    wx=(weight[:,:,:,1:]+weight[:,:,:,:-1])/2
    wy=(weight[:,:,1:,:]+weight[:,:,:-1,:])/2
    tx=((dx.square()+1e-6).sqrt().sub(.001)*wx).mean()/wx.mean().clamp_min(1e-12)
    ty=((dy.square()+1e-6).sqrt().sub(.001)*wy).mean()/wy.mean().clamp_min(1e-12)
    return color,(tx+ty)/2


def total_variation(image):
    """Smooth L1 adjacent-RGB differences; acts on the real model input image."""
    dx=image[:,:,:,1:]-image[:,:,:,:-1]
    dy=image[:,:,1:,:]-image[:,:,:-1,:]
    return ((dx.square()+1e-6).sqrt().sub(.001).mean()
           +(dy.square()+1e-6).sqrt().sub(.001).mean())/2


def logit_color_gain(logits, gain):
    """Separate encoded-luma direction in logits, not a display saturation filter.

    RGB luminance is NOT exactly preserved after the nonlinear pixel mapping.
    """
    if not math.isfinite(gain) or gain<=0:raise ValueError('Positive finite logit chroma gain required')
    if gain==1:return logits
    luma=(logits*logits.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
    return luma+gain*(logits-luma)


def render(parameter, config, cutoff, sigma):
    logits=parameter.rgb_logits(cutoff)
    logits=logit_color_gain(logits,config.get('logit_chroma_gain',1.))
    if config['mapping']=='sigmoid':
        rgb=(logits*config['mapping_gain']).sigmoid()
    elif config['mapping']=='atan':
        rgb=.5+torch.atan(logits*config['mapping_gain'])/math.pi
    else:
        raise ValueError('Unknown pixel mapping')
    return gaussian_blur(rgb,sigma),logits


def border_mask(height,width,device):
    mask=torch.ones(height,width,dtype=torch.bool,device=device)
    y,x=max(1,height//5),max(1,width//5)
    mask[y:-y,x:-x]=False
    return mask


class StopAtTarget(Exception):
    pass


def run(output, changes, fitted_path=None):
    config={**DEFAULTS,**changes}
    if not 0<=config['uniform_mix']<=1 or not 0<config['final_band']<=1:
        raise ValueError('Invalid objective mix or final band')
    if config['steps']<1 or config['unlock_until']<=0:
        raise ValueError('Steps and unlock duration must be positive')
    optimization_schedule(0,config)
    save_steps=set(config['save_steps'])|{config['steps']}
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    model,provenance=load_inceptionv1()
    from lucent.modelzoo.inceptionv1.helper_layers import ReluLayer
    state=RedirectState(last_global_step=10**9,record=True)
    install_redirected_relu(model,state,(ReluLayer,))
    captured={}
    def capture(module,inputs,result):
        captured['value']=result
        raise StopAtTarget()
    handle=dict(model.named_modules())[config['layer']].register_forward_hook(capture)
    def features(image,view_size=0):
        try:
            model(preprocess(model_view(image,view_size)))
        except StopAtTarget:
            pass
        return captured.pop('value')
    matrix,stats=(None,None) if config['color_mode']!='fitted' else load_fitted_color(fitted_path)
    parameter=ScheduledFFT(config['size'],config['std'],'cuda',config['color_mode'],4.,matrix)
    source_hashes={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                   for name in ['rgb_explore.py','positive.py','targets.py','models.py','relu_redirect.py','color_statistics.py','optimize.py']}
    metadata=dict(config=config,target=dict(layer=config['layer'],channel=config['channel'],activation='post-ReLU',
                  aggregation='(1-mix)*Gaussian + mix*spatial mean'),model=provenance,
                  fitted_color=stats,color_matrix=parameter.color_matrix.cpu().tolist(),
                  seed_noise_sha256=parameter.base_noise_sha256,source_hashes=source_hashes,
                  versions=dict(torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__),
                  alpha=False,display_adjustment=False,path_guidance=False,
                  image_sizes=dict(optimized=config['size'],model_input=config['view_size'] or config['size'],
                                   common_evaluation=config['common_view_size'] or config['size']),
                  redirect='until first positive augmented objective, then permanently standard')
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    common=NeuronTarget(config['layer'],config['channel'],'gaussian',.2)
    def evaluate(label):
        # Standard ReLU backward for diagnostics, even at an inactive start.
        saved=state.last_global_step;state.last_global_step=-1
        image,_=render(parameter,config,config['final_band'],config['blur_end'])
        image=image.detach().requires_grad_(True)
        field=features(image,config['view_size'])
        score=spatial_score(field,config)
        grad,=torch.autograd.grad(score,image)
        density=grad.abs().sum(1)[0];edge=border_mask(*density.shape,density.device)
        values=field.detach()[0,config['channel']];fe=border_mask(*values.shape,values.device)
        with torch.no_grad():
            common_size=config['common_view_size'] or config['size']
            training_size=config['view_size'] or config['size']
            common_field=features(image,common_size) if common_size!=training_size else field
            common_score=float(common(common_field).detach())
        result=dict(training_score=float(score.detach()),common_gaussian02=common_score,
            field_mean=float(values.mean()),field_border_mean=float(values[fe].mean()),
            active_fraction=float((values>0).float().mean()),
            input_border_gradient_share=float(density[edge].sum()/density.sum().clamp_min(1e-20)),
            input_border_area_fraction=float(edge.float().mean()))
        save_image(image,output/(label+'.png'))
        np.save(output/(label+'_field.npy'),values.cpu().numpy())
        np.save(output/(label+'_input_gradient.npy'),density.cpu().numpy())
        state.last_global_step=saved
        return result
    initial=evaluate('initial')
    optimizer=torch.optim.Adam(parameter.parameters(),lr=config['lr'])
    first=None;history=[]
    for step in range(config['steps']):
        state.global_step=step;state.last_global_step=step if first is None else -1
        state.reset_counts()
        cutoff,sigma,lr=optimization_schedule(step,config)
        optimizer.param_groups[0]['lr']=lr
        optimizer.zero_grad()
        image,logits=render(parameter,config,cutoff,sigma)
        transformed=augment(image,config)
        field=features(transformed,config['view_size']);score=spatial_score(field,config)
        if first is None and float(score.detach())>0:
            first=step;state.last_global_step=-1
            field=features(transformed,config['view_size']);score=spatial_score(field,config)
        l2=logits.square().mean()
        color=(image.mean((0,2,3))-image.new_tensor(config['mean_color'])).square().mean()
        periphery=periphery_penalty(image,config)
        periphery_color,periphery_tv=periphery_statistics(image,config)
        tv=total_variation(image)
        reward=activation_reward(score,config)
        loss=(-reward+config['logit_l2']*l2+config['mean_color_weight']*color
              +config['periphery_weight']*periphery+config['tv_weight']*tv
              +config['periphery_mean_weight']*periphery_color+config['periphery_tv_weight']*periphery_tv)
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite loss')
        loss.backward()
        if not torch.isfinite(parameter.coefficients.grad).all():
            raise RuntimeError('Nonfinite gradient')
        optimizer.step()
        history.append(dict(step=step,activation=float(score.detach()),loss=float(loss.detach()),
            cutoff=cutoff,sigma=sigma,lr=lr,logit_l2=float(l2.detach()),mean_color_penalty=float(color.detach()),
            activation_reward=float(reward.detach()),
            periphery_penalty=float(periphery.detach()),
            total_variation=float(tv.detach()),
            periphery_mean_penalty=float(periphery_color.detach()),periphery_tv=float(periphery_tv.detach()),
            redirected_ops=state.opened_batch_ops,redirected_elements=state.passed_negative_elements))
        if (step+1) in save_steps:
            with torch.no_grad():
                save_image(render(parameter,config,config['final_band'],config['blur_end'])[0],output/f'checkpoint_{step+1}.png')
        if step%200==0 or step==config['steps']-1:
            print(json.dumps(history[-1]),flush=True)
    final=evaluate('final')
    torch.save(parameter.state_dict(),output/'state.pt')
    with (output/'history.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=history[0]);writer.writeheader();writer.writerows(history)
    (output/'metrics.json').write_text(json.dumps(dict(initial=initial,final=final,first_positive_step=first),indent=2))
    handle.remove()
    print(str(output),flush=True)
