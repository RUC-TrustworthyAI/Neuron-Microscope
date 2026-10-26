"""RelP neuron graphs, adapted from TransluceAI/circuits (pinned in relp-source.json).

Uses their fixed-normalization, frozen QK, SiLU identity and gated half rules.
Native HF forward values and SDPA kernels are retained; only backward is changed.
"""
import contextlib
import types
from pathlib import Path
import torch
from common import write
from settings import load

METHOD = ("RelP: frozen RMS scales and attention QK, SiLU identity rule and gated half rule; "
          "local edges block intervening MLP outputs. Edge weight is root-directed flow; "
          "relative weight and local contribution are saved separately.")
SOURCE_COMMIT = "2d215d4ba016ba8602b69d71fc0f1ad139a427b7"


class FixedScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, value, coefficient):
        ctx.save_for_backward(coefficient)
        return value
    @staticmethod
    def backward(ctx, gradient):
        coefficient, = ctx.saved_tensors
        return (gradient.float() * coefficient.float()).to(gradient.dtype), None, None


class HalfProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return x * y
    @staticmethod
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        return .5 * gradient * y, .5 * gradient * x


@contextlib.contextmanager
def rules(model, local=False):
    """Keep the native forward; local leaves remove indirect MLP-to-MLP paths."""
    state = {'products': {}, 'sources': {}, 'calls': {}, 'embedding': None}
    originals, hooks = [], []
    def replace(module, function):
        originals.append((module, module.forward))
        module.forward = types.MethodType(function, module)
    def norm_forward(original, module):
        def forward(self, x):
            value = original(x)
            c = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + module.variance_epsilon)
            c = c * module.weight.float()
            return FixedScale.apply(x, value.detach(), c.detach())
        return forward
    def silu_forward(original):
        def forward(self, x):
            value = original(x)
            coefficient = (value / (x + 1e-10)).detach()
            coefficient = torch.where(x == 0, torch.full_like(coefficient, .5), coefficient)
            return FixedScale.apply(x, value.detach(), coefficient)
        return forward
    def mlp_forward(self, x):
        gate = self.act_fn(self.gate_proj(x))
        return self.down_proj(HalfProduct.apply(gate, self.up_proj(x)))
    def capture(layer):
        def hook(module, args):
            value = args[0]
            state['products'][layer] = value
            state['calls'][layer] = state['calls'].get(layer, 0) + 1
            source = value.detach().requires_grad_(True) if local else value
            state['sources'][layer] = source
            if local:
                return (source,) + args[1:]
        return hook
    def embed(module, args, output):
        value = output.detach().requires_grad_(True)
        state['embedding'] = value
        return value
    try:
        for module in model.modules():
            if module.__class__.__name__ == 'Qwen3RMSNorm':
                replace(module, norm_forward(module.forward, module))
        for layer, block in enumerate(model.model.layers):
            replace(block.mlp.act_fn, silu_forward(block.mlp.act_fn.forward))
            replace(block.mlp, mlp_forward)
            # Detaching Q/K preserves the original SDPA forward and its OV derivative.
            attention=getattr(block,'self_attn',None)
            for projection in ([attention.q_proj,attention.k_proj] if attention is not None else []):
                hooks.append(projection.register_forward_hook(lambda m, a, out: out.detach()))
            hooks.append(block.mlp.down_proj.register_forward_pre_hook(capture(layer)))
        hooks.append(model.model.embed_tokens.register_forward_hook(embed))
        yield state
    finally:
        for hook in hooks:
            hook.remove()
        for module, forward in reversed(originals):
            module.forward = forward


def selected_nodes(runtime, ids, pos, layer, unit, token_id=None, budget=None, embeddings=None):
    cfg=load()['graphs']
    budget=cfg['mlp_nodes'] if budget is None else budget
    embeddings=cfg['embedding_nodes'] if embeddings is None else embeddings
    with rules(runtime.model) as s, torch.enable_grad():
        logits = runtime.forward(ids, pos)
        root = s['products'][layer][0, pos, unit] if token_id is None else logits[token_id]
        last = layer if token_id is None else len(runtime.model.model.layers) - 1
        sources = [s['sources'][i] for i in range(last+1)] + [s['embedding']]
        gradients = torch.autograd.grad(root, sources, allow_unused=True)
        candidates, coverage = [], []
        for l, gradient in enumerate(gradients[:-1]):
            a = s['products'][l][0, :pos+1].detach().float()
            coverage.append({'layer':l, 'positions':pos+1, 'width':a.shape[-1],
                             'native_calls':s['calls'][l], 'gradient_observed':gradient is not None})
            if gradient is None:
                continue
            g = gradient[0, :pos+1].detach().float()
            scores = (a*g).flatten()
            if l == layer:
                scores[pos*a.shape[-1]+unit] = 0
            k = min(budget, scores.numel())
            ix = scores.abs().topk(k).indices.tolist()
            vals = scores[ix].cpu().tolist()
            av = a.flatten()[ix].cpu().tolist()
            gv = g.flatten()[ix].cpu().tolist()
            for index, score, activation, derivative in zip(ix, vals, av, gv):
                if score:
                    p, n = divmod(index, a.shape[-1])
                    candidates.append({'node_id':f'{l}_{n}_{p}', 'layer':l, 'feature':n,
                        'ctx_idx':p, 'feature_type':'mlp_neuron', 'influence':score,
                        'activation':activation, 'gradient':derivative, 'label':None})
        nodes = sorted(candidates, key=lambda n:-abs(n['influence']))[:budget]
        ea = s['embedding'][0, :pos+1].detach().float()
        eg = gradients[-1][0, :pos+1].detach().float()
        scores = (ea*eg).sum(-1)
        for p in scores.abs().topk(min(embeddings, len(scores))).indices.tolist():
            if float(scores[p]):
                nodes.append({'node_id':f'E_{ids[p]}_{p}', 'layer':'embedding',
                    'feature':ids[p], 'ctx_idx':p, 'feature_type':'embedding',
                    'influence':float(scores[p]), 'activation':1.,
                    'token':runtime.tokenizer.decode([ids[p]])})
        activation = float(s['products'][layer][0, pos, unit].detach())
        grad = gradients[layer]
        derivative = float(grad[0,pos,unit]) if grad is not None else 0.
        target = {'node_id':f'{layer}_{unit}_{pos}', 'layer':layer, 'feature':unit,
                  'ctx_idx':pos, 'feature_type':'mlp_neuron', 'activation':activation,
                  'gradient':derivative, 'influence':activation*derivative,
                  'is_target_neuron':True, 'label':None}
        nodes.append(target)
        logit = None
        if token_id is not None:
            logit = {'node_id':f'L_{token_id}_{pos}', 'layer':'logit',
                'feature':token_id, 'ctx_idx':pos, 'feature_type':'logit',
                'activation':float(root.detach()), 'influence':float(root.detach()),
                'clerp':f'Logit: {runtime.tokenizer.decode([token_id])!r}'}
        return nodes, logit, float(root.detach()), logits.detach().cpu(), coverage


def local_edges(runtime, ids, pos, nodes, logit=None, max_edges=None):
    max_edges=load()['graphs']['max_edges'] if max_edges is None else max_edges
    links = []
    mlp_nodes = [n for n in nodes if isinstance(n['layer'], int)]
    with rules(runtime.model, local=True) as s, torch.enable_grad():
        logits = runtime.forward(ids, pos)
        receivers = mlp_nodes + ([logit] if logit else [])
        for target in receivers:
            layer = target['layer']
            last = len(runtime.model.model.layers) if layer == 'logit' else layer
            candidates = [n for n in nodes if n['layer'] == 'embedding' or
                          (isinstance(n['layer'], int) and n['layer'] < last and n['ctx_idx'] <= target['ctx_idx'])]
            if not candidates:
                continue
            source_layers = sorted({n['layer'] for n in candidates if isinstance(n['layer'], int)})
            sources = [s['sources'][l] for l in source_layers]
            with_embedding = any(n['layer'] == 'embedding' for n in candidates)
            if with_embedding:
                sources.append(s['embedding'])
            root = logits[target['feature']] if layer == 'logit' else s['products'][layer][0,target['ctx_idx'],target['feature']]
            gradients = torch.autograd.grad(root, sources, retain_graph=True, allow_unused=True)
            gs = dict(zip(source_layers, gradients[:len(source_layers)]))
            receiver_value = float(root.detach())
            # Same adaptive epsilon convention as Transluce's relative-attribution edges.
            reference = logits if layer == 'logit' else s['products'][layer]
            epsilon = float(reference.detach().float().abs().mean()) * 1e-6
            denominator = receiver_value + epsilon
            if denominator == 0:
                continue
            for source in candidates:
                p, n, l = source['ctx_idx'], source['feature'], source['layer']
                if l == 'embedding':
                    g = gradients[-1]
                    if g is None:
                        continue
                    contribution = float((s['embedding'][0,p].detach().float()*g[0,p].float()).sum())
                else:
                    g = gs.get(l)
                    if g is None:
                        continue
                    contribution = float((s['sources'][l][0,p,n].detach().float()*g[0,p,n].float()))
                relative = contribution / denominator
                flow = relative * target['influence']
                if flow:
                    links.append({'source':source['node_id'], 'target':target['node_id'],
                        'weight':flow, 'final_attribution':flow,
                        'relative_weight':relative, 'local_contribution':contribution})
    return sorted(links, key=lambda e:-abs(e['weight']))[:max_edges]


def graph(runtime, ids, pos, layer, unit, token_id=None, budget=None, embeddings=None, max_edges=None):
    nodes, logit, root, logits, coverage = selected_nodes(
        runtime, ids, pos, layer, unit, token_id, budget, embeddings)
    links = local_edges(runtime, ids, pos, nodes, logit, max_edges)
    target = next(n for n in nodes if n.get('is_target_neuron'))
    output = {'tokens':[runtime.tokenizer.decode([i]) for i in ids], 'position':pos,
        'target':'logits' if logit else 'neuron', 'activation':target['activation'],
        'nodes':nodes, 'links':links, 'logit_nodes':[logit] if logit else [],
        'neuron_nodes':[target], 'meta':{'method':METHOD, 'root_value':root,
            'root_token_id':token_id, 'input_token_ids':ids,
            'observed_prefix_token_ids':ids[:pos+1], 'following_target_token_ids':ids[pos+1:],
            'upstream_commit':SOURCE_COMMIT, 'edge_quantity':'root-directed flow',
            'selected_mlp_nodes':sum(isinstance(n['layer'],int) for n in nodes),
            'edge_receiver_count':len({e['target'] for e in links})}}
    return output, logits, coverage


def direct(runtime, ids, pos, layer, unit, reader_units=(), top_units=None):
    """Signed local RelP contributions for explicitly selected full-prefix inputs."""
    top_units=load()["atlas"]["top_units"] if top_units is None else top_units
    with rules(runtime.model, local=True) as s, torch.enable_grad():
        runtime.forward(ids)
        root = s['products'][layer][0,pos,unit]
        upstream = []
        if layer:
            gradients = torch.autograd.grad(root, [s['sources'][l] for l in range(layer)],
                                            retain_graph=True, allow_unused=True)
            for l, gradient in enumerate(gradients):
                if gradient is None:
                    continue
                scores = (gradient[0,:pos+1].float()*s['sources'][l][0,:pos+1].detach().float()).sum(0)
                ix = scores.abs().topk(min(top_units, scores.numel())).indices.tolist()
                vals = scores[ix].detach().cpu().tolist()
                upstream.extend({'layer':l,'neuron':n,'weight':v} for n,v in zip(ix,vals) if v)
        upstream = sorted(upstream, key=lambda x:-abs(x['weight']))[:top_units]
        downstream = []
        for reader in reader_units:
            rl, rn = int(reader['layer']), int(reader.get('neuron',reader.get('index')))
            if rl <= layer:
                continue
            # Same-position nominated readers; this bounded coverage is kept in the record.
            target = s['products'][rl][0,pos,rn]
            gradient = torch.autograd.grad(target, s['sources'][layer], retain_graph=True, allow_unused=True)[0]
            value = float(gradient[0,pos,unit].float()*s['sources'][layer][0,pos,unit].detach().float()) if gradient is not None else 0.
            downstream.append({'layer':rl, 'neuron':rn, 'weight':value,
                'activation':float(target.detach()), 'position':pos})
        return upstream, downstream
