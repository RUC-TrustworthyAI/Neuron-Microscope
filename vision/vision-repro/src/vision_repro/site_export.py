"""Export completed batches to the microscope without mixing data with source."""
import hashlib
import json
from pathlib import Path
import shutil

from .catalog import LAYERS, SLOTS, validate_target


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def public_metadata(value):
    """Retain scientific values but remove machine-specific absolute paths."""
    if isinstance(value, dict):
        return {k:public_metadata(v) for k,v in value.items()}
    if isinstance(value, list):
        return [public_metadata(v) for v in value]
    if isinstance(value, str) and value.startswith(('/home/','/Users/','/private/','/tmp/')):
        return Path(value).name
    return value


def within(root, relative):
    path = (root/relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f'Asset escapes its run directory: {relative}')
    return path


def batch_paths(root):
    root = Path(root)
    if (root/'complete.json').is_file() and (root/'metadata.json').is_file():
        return [root]
    return sorted(p.parent for p in (root/'batches').glob('*/complete.json')
                  if (p.parent/'metadata.json').is_file())


def export(output, runs=(), sample_run=None, reference_catalog=None, rf_snapshot=None,
           published_api=False):
    if rf_snapshot and not sample_run:
        raise ValueError('--rf-snapshot requires --sample-run for full-input atlases')
    output = Path(output)
    if output.exists():
        raise FileExistsError('Use a new export directory; existing sites are never overwritten')
    output.mkdir(parents=True)
    template = Path(__file__).parent/'web/microscope'
    for name in ('index.html','app.js','style.css'):
        shutil.copyfile(template/name, output/name)
    layers = [{'name':l,'group':'Convolution' if l.startswith('conv') else 'Inception',
               'channels':c,'grid':g} for l,c,g in LAYERS]
    refs = {'base':'https://already-clean.goodfire.com/microscope/inceptionv1/',
            'neurons':{f'{l}:{ch}':{'flags':'00000'} for l,c,_ in LAYERS for ch in range(c)}}
    if reference_catalog:
        given = read(reference_catalog)
        for key, entry in given['neurons'].items():
            if key not in refs['neurons'] or len(entry.get('flags','')) != len(SLOTS):
                raise ValueError('Reference catalog does not match the InceptionV1 catalog')
            refs['neurons'][key] = {'flags':entry['flags']}
    ours = {}; exported = 0
    for root in runs:
        batches = batch_paths(root)
        if not batches:
            raise ValueError(f'No completed batches found in {root}')
        for folder in batches:
            meta, completion, evaluation = (read(folder/f'{n}.json') for n in ('metadata','complete','evaluation'))
            layer, slot, channels = meta['layer'], meta['slot'], meta['channels']
            validate_target(layer, channels)
            if slot not in SLOTS or completion['executed_steps'] < 0:
                raise ValueError('Invalid completed batch')
            if completion['steps'] != meta['config']['steps'] or 'final' not in evaluation:
                raise ValueError('Incomplete optimization record')
            steps = completion['steps']
            for i, ch in enumerate(channels):
                key = f'{layer}:{ch}'
                node = ours.setdefault(key, {'layer':layer,'channel':ch,'ours':{},'variantInfo':{},
                                              'model':public_metadata(meta['model'])})
                if slot in node['ours']:
                    raise ValueError(f'Duplicate {key}/{slot}; select one run explicitly')
                paths, scores = {}, {}
                labels = [('initial',0), *[(f'checkpoint_{s}',int(s)) for s in evaluation if s.isdigit()], ('final',steps)]
                for label, step in labels:
                    source = folder/f'items/{i}/{label}.png'
                    if not source.is_file():
                        if label in ('initial','final'): raise FileNotFoundError(source)
                        continue
                    rel = f'assets/ours/{layer}/{ch}/{slot}/{step}.png'
                    destination = output/rel; destination.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(source,destination)
                    phase = 'initial' if label == 'initial' else 'final' if label == 'final' else str(step)
                    paths[str(step)] = rel
                    scores[str(step)] = evaluation[phase][i]['score']
                node['ours'][slot] = paths
                node['variantInfo'][slot] = public_metadata({'runId':folder.name,'config':meta['config'],
                    'activation':meta['activation'],'direction':meta['direction'],'members':meta['members'],
                    'scores':scores,'initial':evaluation['initial'][i],'final':evaluation['final'][i],
                    'signed_mapping':meta['signed_mapping'][i], 'source_hashes':meta.get('source_hashes',{})})
                exported += 1
    samples = None
    dataset = None
    if sample_run:
        raw = Path(sample_run)
        web = raw/'web' if (raw/'web').is_dir() else raw
        complete = read(web/'complete.json')
        protocol = read(web/'protocol.json')
        manifest = read(web/'manifest.json')
        digest = hashlib.sha256((web/'manifest.json').read_bytes()).hexdigest()
        if complete.get('manifest_sha256') != digest:
            raise ValueError('Sample manifest checksum mismatch')
        for item in manifest:
            source = within(web,item['path'])
            if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError(f'Sample checksum mismatch: {item["path"]}')
            dest = within(output/'samples',item['path']); dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,dest)
        (output/'samples').mkdir(exist_ok=True)
        write(output/'samples/protocol.json',public_metadata(protocol))
        write(output/'samples/complete.json',complete)
        # complete.json hashes these exact bytes, not a reserialized JSON object.
        shutil.copyfile(web/'manifest.json',output/'samples/manifest.json')
        samples = 'samples/'
        dataset = {'name':'ImageNet train','samples':complete['images_scanned'],
                   'source':'our-model-scan','fullScan':protocol.get('full_imagenet_scan',False)}
    write(output/'reference-catalog.json',refs)
    write(output/'our-results.json',ours)
    rf_examples = None
    if rf_snapshot:
        from .rf_site_export import attach
        rf_examples = attach(output, rf_snapshot)
    data = {'schema':3,'layers':layers,'neuronsCount':sum(c for _,c,_ in LAYERS),
            'referenceBase':refs['base'],'referenceCatalog':'reference-catalog.json','ours':'our-results.json',
            'sampleRoot':samples,'sampleTransport':'raw','dataset':dataset,
            'defaultSteps':1024,'defaultCompare':True,'optimizedNeurons':list(ours),
            'optimizationVariantsExported':exported,'selectedPositiveSetting':'A',
            'rfExamples':rf_examples,'publishedApi':published_api}
    write(output/'data.json',data)
    assets = []
    for path in sorted(output.rglob('*')):
        if path.is_file():
            assets.append({'path':path.relative_to(output).as_posix(),
                           'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size})
    write(output/'asset-manifest.json',assets)
    return {'neurons':data['neuronsCount'],'populated_neurons':len(ours),
            'exported_variants':exported,'natural_samples':bool(samples),'output':str(output)}
