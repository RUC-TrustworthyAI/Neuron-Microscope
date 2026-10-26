"""Shared JSON, token selection and dense-Qwen3 runtime."""
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import time
from pathlib import Path
FRAMEWORK = Path(__file__).resolve().parents[1]

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def read(path):
    return json.loads(Path(path).read_text())

def write(path, obj):
    path = Path(path)
    text = json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2) + '\n'
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(text)
    temp.replace(path)

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def integer(value, name, low=0):
    if type(value) is not int or value < low:
        raise ValueError(f'{name} must be an integer >= {low}')
    return value

def position(value, length):
    if value == -1 and type(value) is int:
        return length - 1
    value = integer(value, 'token position')
    if value >= length:
        raise ValueError('Token position outside input')
    return value

def span_positions(offsets, start, end):
    indices = [i for (i, (a, b)) in enumerate(offsets) if a < end and b > start and (b > a)]
    if not indices:
        raise ValueError('Scored character span has no overlapping input tokens')
    return indices

def score_positions(text, offsets, replacement_span, score):
    location = score['location']
    if location == 'input':
        selected=list(range(len(offsets)))
    elif location == 'after_replacement':
        selected=[i for i,(a,b) in enumerate(offsets) if b>replacement_span[1]]
    elif location == 'replacement_span':
        selected = span_positions(offsets, *replacement_span)
    elif location == 'token_position':
        selected = [position(score['position'], len(offsets))]
    elif location == 'marker':
        marker = score['text']
        if not marker:
            raise ValueError('Empty marker')
        (starts, start) = ([], 0)
        while True:
            i = text.find(marker, start)
            if i < 0:
                break
            starts.append(i)
            start = i + 1
        occurrence = score.get('occurrence')
        if occurrence is None:
            occurrence = 0
        occurrence = integer(occurrence, 'marker occurrence')
        if occurrence >= len(starts):
            raise ValueError('Marker occurrence not found')
        i = starts[occurrence]
        selected = span_positions(offsets, i, i + len(marker))
        offset = score.get('token_offset')
        if offset is not None:
            if type(offset) is not int or offset < -len(selected) or offset >= len(selected):
                raise ValueError('Marker token_offset outside marker tokens')
            selected = [selected[offset]]
    else:
        raise ValueError('Unknown scoring location')
    if score['reduction'] == 'single' and len(selected) != 1:
        raise ValueError('Single-token scoring resolved to multiple tokens')
    return selected

def select_peak(records, reduction):
    if not records:
        raise ValueError('No measured positions')
    if any((not math.isfinite(x['product']) for x in records)):
        raise ValueError('Nonfinite activation')
    if reduction == 'single':
        if len(records) != 1:
            raise ValueError('Expected one measured token')
        return records[0]
    if reduction == 'max':
        return max(records, key=lambda x: (x['product'], -x['position']))
    if reduction == 'min':
        return min(records, key=lambda x: (x['product'], x['position']))
    raise ValueError('Unknown reduction')

def tokenize(tokenizer, text, limit):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False)
    ids = list(encoded['input_ids'])
    if not ids or (limit is not None and len(ids) > limit):
        raise ValueError(f'Input has {len(ids)} tokens; allowed range is 1..{limit}; no truncation')
    return (ids, [list(x) for x in encoded['offset_mapping']])

def model_config(path, neuron):
    path = Path(path).resolve()
    config = read(path / 'config.json')
    if config.get('model_type') != 'qwen3':
        raise ValueError('This adapter supports dense Qwen3 only')
    layer = integer(neuron['layer'], 'layer')
    index = integer(neuron['index'], 'neuron index')
    if layer >= config['num_hidden_layers'] or index >= config['intermediate_size']:
        raise ValueError('Zero-based neuron coordinate outside model')
    return (path, config)

def model_identity(path, config):
    path = Path(path)
    return {'path': str(path), 'model_type': config['model_type'], 'config_sha256': digest(path / 'config.json'), 'small_files': {p.name: digest(p) for p in [path / 'tokenizer.json', path / 'model.safetensors.index.json'] if p.is_file()}, 'shards': [{'name': p.name, 'bytes': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in sorted(path.glob('*.safetensors'))], 'shards_fully_hashed': False}

class Runtime:
    """Model weights stay frozen. No implicit model downloads or device reassignment."""

    def __init__(self, model, tokenizer, settings, neuron):
        import torch
        (self.torch, self.model, self.tokenizer) = (torch, model.eval(), tokenizer)
        (self.settings, self.neuron) = (settings, neuron)
        self.forwards = 0
        self._parameter_storages = {(p.device, p.untyped_storage().data_ptr()) for p in self.model.parameters()}
        for p in self.model.parameters():
            p.requires_grad_(False)

    @classmethod
    def load(cls, request):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        (path, config) = model_config(request['model'], request['neuron'])
        from settings import request_settings
        cfg=request_settings(request)
        settings = dict(cfg['runtime'],devices=list(range(len(cfg['gpu_groups'][0]))))
        settings.update(request.get('runtime', {}))
        torch.set_num_threads(settings['threads'])
        torch.manual_seed(request.get('torch_seed',cfg['torch_seed']))
        torch.backends.cuda.matmul.allow_tf32 = settings['tf32']
        torch.backends.cudnn.allow_tf32 = settings['tf32']
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = settings['bf16_reduced_precision_reduction']
        devices = settings['devices']
        if not torch.cuda.is_available() or any((i >= torch.cuda.device_count() for i in devices)):
            raise ValueError('Requested logical CUDA devices are unavailable')
        dtype = {'bfloat16': torch.bfloat16, 'float32': torch.float32}[settings['dtype']]
        count = config['num_hidden_layers']
        mapping = {'model.embed_tokens': devices[0], 'model.norm': devices[-1], 'lm_head': devices[-1]}
        mapping.update({f'model.layers.{i}': devices[min(len(devices) - 1, i * len(devices) // count)] for i in range(count)})
        torch.cuda.init()
        for device in devices:
            with torch.cuda.device(device):
                torch.empty(0, device='cuda:' + str(device))
                torch.cuda.reset_peak_memory_stats(device)
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True, use_fast=True)
        if not tokenizer.is_fast:
            raise ValueError('Offset-aware fast tokenizer is required')
        before = model_identity(path, config)
        model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, dtype=dtype, attn_implementation=settings['attention'], device_map=mapping).eval()
        result = cls(model, tokenizer, settings, request['neuron'])
        result.identity = model_identity(path, config)
        result.identity['metadata_before_load'] = before
        result.versions = {'torch': torch.__version__, 'transformers': transformers.__version__, 'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        return result

    def forward(self, ids, logit_position=None):
        torch = self.torch
        tensor = torch.tensor([ids], dtype=torch.long, device=self.model.model.embed_tokens.weight.device)
        # Keep weights on the selected GPUs; offload only tensors saved for backward.
        # CPU copies retain dtype and values. No input truncation or arithmetic change.
        def pack(value):
            key=(value.device,value.untyped_storage().data_ptr())
            saved=value.detach()
            # SDPA backward consumes saved strides, including singleton axes.
            # CPU round trips may compact padded or broadcast views. Keep those
            # views on their original device; offload only canonical dense bases.
            stride = 1
            canonical = value.storage_offset() == 0
            for size, actual in zip(reversed(value.shape), reversed(value.stride())):
                canonical = canonical and actual == stride
                stride *= max(size, 1)
            if value.is_cuda and key not in self._parameter_storages and canonical:
                saved=saved.to('cpu')
            return value.device,saved
        def unpack(saved):
            device,value=saved
            return value.to(device)
        offload=self.settings.get('activation_offload')=='cpu' and torch.is_grad_enabled()
        context=torch.autograd.graph.saved_tensors_hooks(pack,unpack) if offload else contextlib.nullcontext()
        with context:
            hidden = self.model.model(input_ids=tensor, use_cache=False).last_hidden_state
            self.forwards += 1
            return None if logit_position is None else self.model.lm_head(hidden[:, logit_position:logit_position + 1])[0, 0].float()

    @contextlib.contextmanager
    def capture(self, zero_position=None, scale=0., neuron_index=None):
        (state, handles) = ({'calls': {'gate': 0, 'value': 0, 'product': 0}}, [])
        mlp = self.model.model.layers[self.neuron['layer']].mlp
        n = self.neuron['index'] if neuron_index is None else neuron_index

        def capture(name):

            def hook(module, args, out):
                state['calls'][name] += 1
                state[name] = out[0, :, n].detach().float().cpu()
            return hook

        def down(module, args):
            x = args[0]
            if x.ndim != 3 or x.shape[0] != 1:
                raise ValueError('Expected unpadded batch-one native MLP input')
            state['calls']['product'] += 1
            state['product'] = x[0, :, n].detach().float().cpu()
            if zero_position is not None:
                y = x.clone()
                before = float(y[0, zero_position, n])
                y[0, zero_position, n] *= scale
                state['edit'] = {'position': zero_position, 'neuron': n, 'before': before, 'after': float(y[0, zero_position, n]), 'multiplier':scale, 'entries_targeted': 1, 'entries_changed': int(before != 0)}
                return (y,) + args[1:]
        try:
            handles.append(mlp.gate_proj.register_forward_hook(capture('gate')))
            handles.append(mlp.up_proj.register_forward_hook(capture('value')))
            handles.append(mlp.down_proj.register_forward_pre_hook(down))
            yield state
        finally:
            for h in handles:
                h.remove()

    def telemetry(self):
        torch = self.torch
        return {'forwards': self.forwards, 'settings': self.settings, 'versions': getattr(self, 'versions', {}), 'cuda_memory': [{'logical_device': d, 'peak_allocated_bytes': torch.cuda.max_memory_allocated(d), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(d)} for d in self.settings.get('devices', []) if torch.cuda.is_available()]}
