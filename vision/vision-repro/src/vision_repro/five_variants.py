"""Five-slot feature visualization: explicit signed poles, path anchors and diversity.

This is an implementation of the public method descriptions, not recovered
Goodfire dream02 code. The pooling, loss scales and 50k anchor subset are stated
choices. Images are native 224px; no image-generation model or display recoloring.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .color_statistics import load_fitted_color
from .local_color import LocalColorLoss, neighbor_rgb_loss, patch_pair_loss, validate_local_color
from .models import load_inceptionv1, preprocess, PROJECT_ROOT
from .optimize import augment, save_image
from .positive import ScheduledFFT, gaussian_blur
from .relu_redirect import RedirectState, install_redirected_relu
from .rgb_explore import DEFAULTS, optimization_schedule, render, periphery_statistics

CHANNELS = (447, 226, 212)
BRANCHES = ('1x1', '3x3', '5x5', 'pool_reduce')
SLOTS = ('positive', 'path_positive', 'diversity', 'negative', 'path_negative')
BASE_CONFIG = {**DEFAULTS, 'size': 224, 'view_size': 224, 'common_view_size': 224,
    'steps': 2048, 'schedule_steps': 768, 'seed': 0, 'color_mode': 'fitted',
    'final_band': .45, 'gaussian_sigma': .12, 'periphery_power': 2,
    'periphery_mean_weight': 80000, 'mean_color_weight': 20000,
    'mean_color': [.65,.70,.74], 'save_steps': [64,192,384,768,1024,1536,2048],
    'path_steps': 64, 'path_weight_ratio': 1., 'diversity_weight_ratio': .3}


def appearance_regularizers(image, config):
    """Experimental encoded-RGB luma contrast / chroma high-pass penalties.

    Neither term compares against reference pixels. Both change the actual
    optimization, never the displayed image. Zero defaults preserve old runs.
    """
    luma=(image*image.new_tensor([.2126,.7152,.0722])[None,:,None,None]).sum(1,keepdim=True)
    std=luma.flatten(1).std(dim=1,unbiased=False)
    contrast=(std-config.get('luma_std_limit',.18)).clamp_min(0).square().mean()
    chroma=image-luma
    residual=chroma-gaussian_blur(chroma,config.get('chroma_blur_sigma',1.5))
    detail=residual.square().mean()
    penalty=config.get('luma_contrast_weight',0.)*contrast+config.get('chroma_detail_weight',0.)*detail
    return penalty,contrast,detail


def match_initial_color(parameter, source_matrix):
    """Change FFT coordinates so initial rendered RGB matches the source basis.

    Frequency scaling/masks are channel-independent, so the same linear color
    change can be applied to the real and imaginary Fourier coefficients.
    This preserves the initial image, not iid coefficient variance in RGB axes.
    """
    with torch.no_grad():
        source = torch.as_tensor(source_matrix, device=parameter.coefficients.device,
                                 dtype=parameter.coefficients.dtype)
        transform = torch.linalg.solve(parameter.color_matrix, source)
        parameter.coefficients.copy_(torch.einsum(
            'ij,bjhwq->bihwq', transform, parameter.coefficients))


def sha256(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def gaussian_weights(field, sigma=.12, centers=None):
    """B×H×W, normalized independently, coordinates [-.5,.5]."""
    b,_,h,w=field.shape
    y=torch.linspace(-.5,.5,h,device=field.device,dtype=field.dtype)
    x=torch.linspace(-.5,.5,w,device=field.device,dtype=field.dtype)
    if centers is None:
        centers=field.new_zeros((b,2))
    else:
        centers=torch.as_tensor(centers,device=field.device,dtype=field.dtype)
    r2=(y[None,:,None]-centers[:,0,None,None]).square()+(x[None,None,:]-centers[:,1,None,None]).square()
    weight=torch.exp(-r2/(2*sigma*sigma))
    return weight/weight.sum((-2,-1),keepdim=True).clamp_min(1e-20)


def channel_scores(field, channel, sigma=.12):
    return (field[:,channel]*gaussian_weights(field,sigma)).sum((-2,-1))


def path_vector(previous, target, channel, sigma=.12, centers=None):
    """Explicit activation-path proxy, NOT pixel gradients / gradient×activation.

    Pool each layer around a spatial center, zero the target channel so its own
    magnitude cannot dominate, normalize each layer, concatenate and normalize.
    Natural anchors use each exemplar's winning location; synthesis uses center.
    """
    a=(previous*gaussian_weights(previous,sigma,centers)[:,None]).sum((-2,-1))
    b=(target*gaussian_weights(target,sigma,centers)[:,None]).sum((-2,-1))
    mask=torch.ones_like(b);mask[:,channel]=0
    b=b*mask
    return F.normalize(torch.cat((F.normalize(a,dim=1),F.normalize(b,dim=1)),1),dim=1)


def unit_mean_anchor(vectors):
    return F.normalize(F.normalize(vectors,dim=1).mean(0),dim=0)


def diversity_penalty(previous, sigma=.12):
    """Mean off-diagonal cosine between previous-layer weighted Gram matrices."""
    b,c,h,w=previous.shape
    if b<2:
        return previous.sum()*0
    x=previous.flatten(2)*gaussian_weights(previous,sigma).flatten(1)[:,None].sqrt()
    gram=torch.bmm(x,x.transpose(1,2)).flatten(1)
    unit=F.normalize(gram,dim=1)
    cosine=unit@unit.T
    return cosine[~torch.eye(b,dtype=torch.bool,device=previous.device)].mean()


class StopAtMixed4c(Exception):
    pass


class Mixed4cCapture:
    """Capture post-ReLU block and exact concatenated signed branch outputs."""
    def __init__(self, model):
        self.model=model;self.values={};self.handles=[]
        modules=dict(model.named_modules())
        self.widths=[modules[f'mixed4c_{b}_pre_relu_conv'].out_channels for b in BRANCHES]
        for name in ['mixed4b',*[f'mixed4c_{b}_pre_relu_conv' for b in BRANCHES],'mixed4c']:
            def hook(module,inputs,result,name=name):
                self.values[name]=result
                if name=='mixed4c':raise StopAtMixed4c()
            self.handles.append(modules[name].register_forward_hook(hook))

    def __call__(self, image):
        self.values={}
        try:self.model(preprocess(image))
        except StopAtMixed4c:pass
        previous=self.values['mixed4b'];post=self.values['mixed4c']
        pre=torch.cat([self.values[f'mixed4c_{b}_pre_relu_conv'] for b in BRANCHES],dim=1)
        return previous,post,pre

    def mapping(self, channel):
        offset=0
        for branch,width in zip(BRANCHES,self.widths):
            if offset<=channel<offset+width:
                return {'node':f'mixed4c_{branch}_pre_relu_conv','channel':channel-offset}
            offset+=width
        raise ValueError('Channel outside mixed4c')

    def close(self):
        for h in self.handles:h.remove()
        self.values={}


def reweight_fft_scale(scale, power):
    if not .5<=power<=3:
        raise ValueError('FFT decay power must be in [.5,3]')
    if power==1:
        return scale
    weighted=scale.pow(power)
    return weighted*(scale.square().mean()/weighted.square().mean()).sqrt()


class BatchFFT(torch.nn.Module):
    def __init__(self, count, config, matrix):
        super().__init__()
        self.images=torch.nn.ModuleList([ScheduledFFT(config['size'],config['std'],'cuda',config['color_mode'],4.,matrix) for _ in range(count)])
        for parameter in self.images:
            parameter.scale.copy_(reweight_fft_scale(parameter.scale,config.get('fft_decay_power',1.)))
            if config.get('match_fitted_initial_rgb', False):
                match_initial_color(parameter, matrix)

    def forward(self, config, cutoff, sigma):
        parts=[render(p,config,cutoff,sigma) for p in self.images]
        return torch.cat([x[0] for x in parts]),torch.cat([x[1] for x in parts])


def save_batch(image, directory, label):
    # Exact 2×2 juxtaposition, no resampling, no color adjustment.
    if image.shape[0]==1:
        save_image(image,directory/f'{label}.png');return
    side=image.shape[-1]
    grid=Image.new('RGB',(2*side,2*side))
    for i in range(image.shape[0]):
        member=directory/f'{label}_member{i}.png';save_image(image[i:i+1],member)
        with Image.open(member) as im:grid.paste(im,((i%2)*side,(i//2)*side))
    grid.save(directory/f'{label}.png')


def run_variant(output, channel, slot, fitted_path, anchor_file=None, changes=None):
    if slot not in SLOTS:raise ValueError(slot)
    config={**BASE_CONFIG,**(changes or {}),'channel':channel,'slot':slot}
    neighbor_weight = config.get('neighbor_rgb_weight', 0.)
    if not math.isfinite(neighbor_weight) or neighbor_weight < 0:
        raise ValueError('neighbor_rgb_weight must be finite and nonnegative')
    validate_local_color(config)
    memory_limit_mb = config.get('cuda_memory_limit_mb')
    if memory_limit_mb is not None and (type(memory_limit_mb) is not int or memory_limit_mb < 1):
        raise ValueError('cuda_memory_limit_mb must be a positive integer')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    if memory_limit_mb is not None:
        total_memory = torch.cuda.get_device_properties('cuda').total_memory
        fraction = memory_limit_mb * 1024**2 / total_memory
        if fraction > 1:
            raise ValueError('CUDA memory limit exceeds device memory')
        torch.cuda.set_per_process_memory_fraction(fraction)
    torch.cuda.reset_peak_memory_stats()
    random.seed(config['seed']);np.random.seed(config['seed']);torch.manual_seed(config['seed'])
    count=4 if slot=='diversity' else 1
    eot=config.get('eot_views',1)
    if not isinstance(eot,int) or not 1<=eot<=8 or (count>1 and eot!=1):
        raise ValueError('EOT supports 1..8 transforms of one image; diversity retains its original batch')
    negative=slot in ('negative','path_negative')
    guided=slot in ('path_positive','path_negative')
    model,provenance=load_inceptionv1()
    from lucent.modelzoo.inceptionv1.helper_layers import ReluLayer
    state=RedirectState(last_global_step=-1,record=True)
    install_redirected_relu(model,state,(ReluLayer,))
    capture=Mixed4cCapture(model)
    matrix,fitted=load_fitted_color(fitted_path)
    local_regularizer = LocalColorLoss(config, matrix).to('cuda')
    legacy_rgb_edges = local_regularizer.space == 'rgb' and local_regularizer.neighborhood == 'edges'
    parameter=BatchFFT(count,config,matrix)
    anchor=None;reference_scale=1.;anchor_info=None
    if anchor_file:
        all_anchors=torch.load(anchor_file,map_location='cpu',weights_only=True)
        key=f'{channel}_{"negative" if negative else "positive"}'
        entry=all_anchors[key]
        anchor=entry['vector'].to('cuda');reference_scale=entry['score_scale']
        anchor_info={'file':str(anchor_file),'sha256':sha256(anchor_file),'key':key,'score_scale':reference_scale}
    if guided and anchor is None:raise ValueError('Path guidance requires a real natural-image anchor')
    if slot=='diversity' and not anchor_file:raise ValueError('Diversity weight is calibrated to the natural score scale')
    path_weight=config['path_weight_ratio']*reference_scale if guided else 0.
    diversity_weight=config['diversity_weight_ratio']*reference_scale if slot=='diversity' else 0.
    source_names=['five_variants.py','local_color.py','natural_anchors.py','rgb_explore.py','positive.py','relu_redirect.py','models.py','targets.py','optimize.py','color_statistics.py']
    metadata={'config':config,'batch_size':count,'model':provenance,
        'target':{'layer':'mixed4c','channel':channel,'activation':'pre-ReLU' if negative else 'post-ReLU','aggregation':'normalized Gaussian, coordinates [-.5,.5]','gaussian_sigma':config['gaussian_sigma'],'direction':'minimize' if negative else 'maximize','signed_branch':capture.mapping(channel)},
        'previous_layer':'mixed4b','fitted_color':fitted,
        'color_matrix':parameter.images[0].color_matrix.detach().cpu().tolist(),
        'color_initialization':{'matched_fitted_rgb':config.get('match_fitted_initial_rgb',False),
            'definition':'Same seed latent noise; optionally map Fourier coefficients into the chosen color coordinates to match fitted-basis initial RGB. No old optimized image is reused.'},
        'source_hashes':{p:sha256(Path(__file__).with_name(p)) for p in source_names if Path(__file__).with_name(p).exists()},
        'seed_noise_sha256':[p.base_noise_sha256 for p in parameter.images],
        'versions':{'torch':torch.__version__,'cuda':torch.version.cuda,'numpy':np.__version__},
        'resource_limits':{'torch_threads':4,'cuda_allocator_limit_mib':memory_limit_mb,
            'scope':'PyTorch caching allocator only; driver/library allocations are not a hardware isolation guarantee.'},
        'path_guidance':{'enabled':guided,'steps':config['path_steps'] if guided else 0,'weight':path_weight,'definition':'cosine to unit-mean natural anchor of separately unit-normalized Gaussian-pooled mixed4b and mixed4c activation vectors; mixed4c target channel zeroed; natural winning-location / synthetic center','anchor':anchor_info if guided else None},
        'diversity':{'enabled':slot=='diversity','weight':diversity_weight,'definition':'mean off-diagonal pairwise cosine between mixed4b Gaussian-weighted Gram matrices','batch':count},
        'redirect':'disabled for negative; for positive until every batch member has positive objective',
        'limitations':['Public-description-inspired, not recovered author code or exact settings.','Natural anchor pool is our class-balanced 50k subset, including for negative; not the author full 1.28M negative-tail scan.','Hand-chosen global/peripheral RGB mean priors remain in all variants.',f'Single run with seed {config["seed"]}; no best-of-four selection for path-negative.','A fixed seed does not imply bitwise deterministic CUDA optimization.'],
        'reproducibility':{'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
            'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark},
        'created_at':datetime.now(timezone.utc).isoformat()}
    metadata['experimental_extensions']={
        'eot_views':eot,'eot_definition':'average objective over independent affine views of the same optimized image; not independent seeds',
        'fft_decay_power':config.get('fft_decay_power',1.),'fft_scale_normalization':'preserve coefficient-grid mean squared spectral scaling before band mask',
        'appearance_regularizers':'one-sided encoded-luma std excess squared; chroma high-pass residual squared. Applied to actual input, never display recoloring.',
        'neighbor_rgb':{'weight':neighbor_weight if legacy_rgb_edges else 0.,'radius_pixels':1,
            'definition':'Universal RGB edge MSE diagnostic; weighted only when RGB edge loss is the active local regularizer.'},
        'local_color':{'weight':neighbor_weight, **local_regularizer.description}}
    write_json(output/'metadata.json',metadata)

    def evaluate(label):
        saved=state.last_global_step;state.last_global_step=-1
        with torch.no_grad():
            image,_=parameter(config,config['final_band'],config['blur_end'])
            previous,post,pre=capture(image)
            # Assertion protects the branch offset, order and nonlinearity mapping.
            if not torch.allclose(post,pre.relu(),atol=1e-4,rtol=1e-5):raise AssertionError('pre/post branch mapping mismatch')
            post_scores=channel_scores(post,channel,config['gaussian_sigma'])
            pre_scores=channel_scores(pre,channel,config['gaussian_sigma'])
            field=pre if negative else post
            scores=pre_scores if negative else post_scores
            cosine=F.cosine_similarity(path_vector(previous,post,channel,config['gaussian_sigma']),anchor[None],dim=1) if anchor is not None else None
            save_batch(image,output,label)
            result={'score':scores.cpu().tolist(),'post_relu':post_scores.cpu().tolist(),'pre_relu':pre_scores.cpu().tolist(),'path_cosine':cosine.cpu().tolist() if cosine is not None else None,'diversity_cosine':float(diversity_penalty(previous,config['gaussian_sigma'])),'same_definition':'unaugmented rendered RGB, final band/blur, native224, standard ReLU, normalized Gaussian sigma .12'}
            result['neighbor_rgb_mse'] = float(neighbor_rgb_loss(image))
            result['neighbor_rgb_weighted'] = neighbor_weight * result['neighbor_rgb_mse'] if legacy_rgb_edges else 0.
            result['patch4_rgb_mse'] = float(patch_pair_loss(image, 4))
            result['local_color_penalty'] = float(local_regularizer(image))
            result['local_color_weighted'] = neighbor_weight * result['local_color_penalty']
            result['global_mean_color_mse'] = float((image.mean((2,3))-image.new_tensor(config['mean_color'])[None]).square().mean())
            result['rgb_mean'] = image.mean((0,2,3)).cpu().tolist()
            np.save(output/f'{label}_field.npy',field[:,channel].cpu().numpy())
        state.last_global_step=saved
        return result

    initial=evaluate('initial');history=[];first=None;checkpoint_metrics={}
    optimizer=torch.optim.Adam(parameter.parameters(),lr=config['lr'])
    started=time.monotonic()
    for step in range(config['steps']):
        state.global_step=step;state.last_global_step=step if not negative and first is None else -1
        state.reset_counts();cutoff,sigma,lr=optimization_schedule(step,config)
        optimizer.param_groups[0]['lr']=lr;optimizer.zero_grad()
        image,logits=parameter(config,cutoff,sigma)
        # Independent geometric draws for each member; never duplicate a member.
        transformed=torch.cat([augment(x[None],config) for x in image for _ in range(eot)])
        previous,post,pre=capture(transformed)
        scores=channel_scores(pre if negative else post,channel,config['gaussian_sigma'])
        if not negative and first is None and bool((scores.detach().reshape(count,eot).mean(1)>0).all()):
            first=step;state.last_global_step=-1
            previous,post,pre=capture(transformed)
            scores=channel_scores(post,channel,config['gaussian_sigma'])
        direction=1 if negative else -1
        path_loss=image.new_tensor(0.)
        if guided and step<config['path_steps']:
            path_loss=(1-F.cosine_similarity(path_vector(previous,post,channel,config['gaussian_sigma']),anchor[None],dim=1)).mean()
        div=diversity_penalty(previous,config['gaussian_sigma']) if count>1 else image.new_tensor(0.)
        color=(image.mean((2,3))-image.new_tensor(config['mean_color'])[None]).square().mean()
        peripheral,_=periphery_statistics(image,config)
        appearance,contrast,chroma_detail=appearance_regularizers(image,config)
        loss=direction*scores.mean()+path_weight*path_loss+diversity_weight*div+config['mean_color_weight']*color+config['periphery_mean_weight']*peripheral+appearance
        # Preserve the zero-weight baseline's original autograd graph.
        local_color = local_regularizer(image if neighbor_weight else image.detach())
        if neighbor_weight:
            loss = loss + neighbor_weight * local_color
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward()
        if not all(p.coefficients.grad is not None and torch.isfinite(p.coefficients.grad).all() for p in parameter.images):raise RuntimeError('Nonfinite/missing FFT gradient')
        optimizer.step()
        row={'step':step+1,'score_mean':float(scores.detach().mean()),'scores':scores.detach().cpu().tolist(),'loss':float(loss.detach()),'path_loss':float(path_loss.detach()),'path_active':bool(guided and step<config['path_steps']),'diversity_cosine':float(div.detach()),'mean_color_penalty':float(color.detach()),'periphery_mean_penalty':float(peripheral.detach()),'cutoff':cutoff,'sigma':sigma,'lr':lr,'redirected_elements':state.passed_negative_elements}
        row.update(appearance_penalty=float(appearance.detach()),luma_contrast_penalty=float(contrast.detach()),chroma_detail_penalty=float(chroma_detail.detach()))
        rgb_neighbor = float(local_color.detach()) if legacy_rgb_edges else float(neighbor_rgb_loss(image.detach()))
        row.update(neighbor_rgb_mse=rgb_neighbor,
                   neighbor_rgb_weighted=neighbor_weight*rgb_neighbor if legacy_rgb_edges else 0.,
                   local_color_penalty=float(local_color.detach()),
                   local_color_weighted=neighbor_weight*float(local_color.detach()))
        history.append(row)
        if step+1 in set(config['save_steps'])|{config['steps']}:
            checkpoint_metrics[str(step+1)]=evaluate(f'checkpoint_{step+1}')
            write_json(output/'checkpoint_metrics.json',checkpoint_metrics)
        if (step+1)%256==0 or step==0 or step==config['steps']-1:
            print(json.dumps({'channel':channel,'slot':slot,**row,'elapsed_seconds':time.monotonic()-started}),flush=True)
    final=evaluate('final')
    torch.save(parameter.state_dict(),output/'state.pt')
    with (output/'history.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=history[0]);writer.writeheader();writer.writerows(history)
    result={'initial':initial,'final':final,'first_all_positive_step':first,'elapsed_seconds':time.monotonic()-started}
    result['gpu_memory'] = {'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                            'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,
                            'allocator_limit_mib':memory_limit_mb}
    write_json(output/'metrics.json',result);capture.close()
    del model,parameter;torch.cuda.empty_cache()
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--anchors',type=Path)
    p.add_argument('--fitted-color',type=Path,default=Path(__file__).parent/'defaults/imagenet_color.json')
    p.add_argument('--steps',type=int,default=2048);p.add_argument('--channels',type=int,nargs='+',default=list(CHANNELS))
    p.add_argument('--slots',nargs='+',choices=SLOTS,default=['negative','path_positive','diversity','path_negative'])
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    write_json(args.output/'protocol.json',{'base_config':BASE_CONFIG,'steps':args.steps,'channels':args.channels,'slots':args.slots,'anchors':str(args.anchors),'positive':'Reuse prior native224 2048-step result for website; do not relabel it as newly computed.'})
    results=[]
    for channel in args.channels:
        for slot in args.slots:
            result=run_variant(args.output/f'mixed4c_{channel}'/slot,channel,slot,args.fitted_color,args.anchors,{'steps':args.steps,'schedule_steps':min(768,args.steps),'path_steps':min(64,args.steps)})
            results.append({'channel':channel,'slot':slot,**result})
            write_json(args.output/'progress.json',{'completed':len(results),'total':len(args.channels)*len(args.slots),'results':results})
    write_json(args.output/'complete.json',{'results':results})


if __name__=='__main__':main()
