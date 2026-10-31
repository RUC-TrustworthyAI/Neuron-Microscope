"""Portable entry points. Dataset, model and hosting downloads are never implicit."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from .catalog import LAYERS, SLOTS, validate_target


def parser():
    p = argparse.ArgumentParser(prog='vision-microscope', description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    sub.add_parser('catalog', help='List 12 layers and all 5,804 channels')
    d = sub.add_parser('doctor', help='Check externally provisioned model assets')
    d.add_argument('--model-dir', type=Path)
    o = sub.add_parser('optimize', help='Optimize one independent batch')
    o.add_argument('--layer', required=True, choices=[x[0] for x in LAYERS])
    o.add_argument('--neurons', type=int, nargs='+', required=True)
    o.add_argument('--variant', choices=SLOTS, default='positive')
    o.add_argument('--steps', type=int, default=1024)
    o.add_argument('--seed', type=int, default=0)
    o.add_argument('--output', type=Path, required=True)
    o.add_argument('--anchors', type=Path, help='Trusted per-layer anchor file; required for path/diversity')
    o.add_argument('--fitted-color', type=Path)
    s = sub.add_parser('scan', help='Scan an existing licensed ImageNet training archive')
    s.add_argument('--archive', type=Path, required=True)
    s.add_argument('--output', type=Path, required=True)
    s.add_argument('--batch-size', type=int, default=24)
    s.add_argument('--workers', type=int, default=4)
    s.add_argument('--seed', type=int, default=20261026)
    a = sub.add_parser('anchors', help='Build natural path anchors from a completed 50k scan')
    a.add_argument('--scan', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    a.add_argument('--batch-size', type=int, default=24)
    c = sub.add_parser('campaign', help='Resumable 5,804 × five-variant campaign')
    c.add_argument('action', choices=['init','run','status'])
    c.add_argument('--root', type=Path, required=True)
    c.add_argument('--batch-images', type=int, default=8)
    c.add_argument('--workers', type=int, default=1)
    e = sub.add_parser('export', help='Build a local static viewer; no upload')
    e.add_argument('--runs', type=Path, nargs='*', default=[])
    e.add_argument('--sample-run', type=Path)
    e.add_argument('--reference-catalog', type=Path)
    e.add_argument('--rf-snapshot', type=Path, help='Verified RF snapshot from crop-examples')
    e.add_argument('--published-api', action='store_true', help='Enable same-origin hosting API')
    e.add_argument('--output', type=Path, required=True)
    rf = sub.add_parser('crop-examples', help='CPU-only RF crops from an existing sample scan')
    rf.add_argument('--scan', type=Path, required=True)
    rf.add_argument('--model-source', type=Path, required=True, help='Pinned Lucent InceptionV1.py')
    rf.add_argument('--output', type=Path, required=True)
    rf.add_argument('--workers', type=int, default=2)
    rf.add_argument('--targets', nargs='+', help='Optional layer:channel subset')
    return p


def main(argv=None):
    p = parser(); a = p.parse_args(argv)
    if a.command == 'catalog':
        print(json.dumps({'neurons':sum(c for _,c,_ in LAYERS), 'layers':[
            {'name':l,'channels':c,'grid':g} for l,c,g in LAYERS]}, indent=2)); return
    if a.command == 'doctor':
        directory = a.model_dir or Path(os.environ.get('VNI_MODEL_DIR', 'source/inceptionv1'))
        files = [directory/'manifest.json', directory/'weights/inception5h.pth', directory/'lucent']
        missing = [str(x) for x in files if not x.exists()]
        if missing: p.error('Missing model assets: '+', '.join(missing))
        with files[1].open('rb') as f: digest = hashlib.file_digest(f, 'sha256').hexdigest()
        manifest = json.loads(files[0].read_text())
        if digest != manifest['weights_sha256']: p.error('Model weight SHA-256 mismatch')
        commit = subprocess.check_output(['git','-C',str(files[2]),'rev-parse','HEAD'],text=True).strip()
        if commit != manifest['source_commit']: p.error('Lucent source commit does not match manifest')
        if subprocess.check_output(['git','-C',str(files[2]),'status','--porcelain','--untracked-files=no'],text=True).strip():
            p.error('Lucent checkout has modified tracked files')
        print(json.dumps({'weights_sha256':digest,'source_commit':manifest['source_commit'],'assets':'OK'})); return
    if a.command == 'optimize':
        try: validate_target(a.layer, a.neurons)
        except ValueError as error: p.error(str(error))
        if a.steps <= 0: p.error('--steps must be positive')
        if a.variant in ('path_positive','path_negative','diversity') and a.anchors is None:
            p.error('This variant requires --anchors from your natural-image scan')
        import torch
        from .full_feature_viz import BatchRunner, config_A, DEFAULT_COLOR
        anchors = torch.load(a.anchors, map_location='cpu', weights_only=True) if a.anchors else None
        cfg = config_A(a.steps); cfg['seed'] = a.seed
        BatchRunner(a.fitted_color or DEFAULT_COLOR).run(a.output,a.layer,a.neurons,a.variant,cfg,anchors)
    elif a.command == 'scan':
        from .all_neuron_examples import build
        build(a.output,a.archive,batch_size=a.batch_size,workers=a.workers,seed=a.seed)
    elif a.command == 'anchors':
        from .full_path_anchors import build
        build(a.output,a.scan,batch_size=a.batch_size)
    elif a.command == 'campaign':
        from .full_feature_campaign import initialize, status
        a.root = a.root.resolve(); a.root.mkdir(parents=True,exist_ok=True)
        if a.action == 'init': initialize(a.root,a.batch_images)
        elif a.action == 'status': print(json.dumps(status(a.root),indent=2))
        else:
            if a.workers < 1: p.error('--workers must be positive')
            from .full_feature_supervisor import supervise
            supervise(a.root,a.workers)
    elif a.command == 'export':
        from .site_export import export
        print(json.dumps(export(a.output,a.runs,a.sample_run,a.reference_catalog,
                                a.rf_snapshot,a.published_api),indent=2))
    elif a.command == 'crop-examples':
        from .prepare_rf_site_snapshot import main as crop
        arguments = [str(a.scan),str(a.output),str(a.model_source),'--workers',str(a.workers)]
        if a.targets: arguments += ['--targets',*a.targets]
        crop(arguments)


if __name__ == '__main__':
    main()
