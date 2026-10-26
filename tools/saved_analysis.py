"""CPU-only analysis of saved native activations and unmeasured test proposals."""
import collections
import copy
import datetime
import importlib.util
import json
import mmap
from pathlib import Path
import numpy as np
from common import read, write, now


def sampled_products(dataset, neurons, token_indices):
    indices = np.asarray(token_indices, dtype=np.int64)
    result = np.empty((len(neurons), len(indices)), dtype=np.float32)
    for part in dataset.manifest['parts']:
        members = [(i, n['part_column']) for i, n in enumerate(neurons) if n['part'] == part['id']]
        if not members:
            continue
        for shard in part['shards']:
            slots = np.flatnonzero((indices >= shard['token_start']) &
                                   (indices < shard['token_start'] + shard['token_count']))
            if not len(slots):
                continue
            path = dataset.root/part['path']/shard['path']
            array = np.load(path, mmap_mode='r', allow_pickle=False)
            try:
                array._mmap.madvise(mmap.MADV_RANDOM)
                columns = np.asarray([c for _, c in members])
                bits = np.array(array[columns[:, None], (indices[slots]-shard['token_start'])[None, :], 2])
                values = (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
                result[np.asarray([i for i, _ in members])[:, None], slots[None, :]] = values
            finally:
                array._mmap.close()
    return result


def prepare(session):
    folder = session.root/'saved-analysis'
    if (folder/'result.json').is_file():
        return read(folder/'result.json')
    folder.mkdir(exist_ok=True)
    data = read(Path(session.config['report'])/'report.json')
    directory = Path(session.config['activations'])
    from dataset_reader import open_dataset
    ds = open_dataset(directory)
    target = 'L%d_N%d' % (session.config['neuron']['layer'], session.config['neuron']['index'])
    try:
        members = {str(x) for ids in data['examples'].values() for x in ids}
        members.update(str(x) for band in data['bands'] for x in band['points'])
        uniform = [str(x) for x in data['scatter'] if str(x) not in members]
        # These are the uniform scatter tokens, excluding the deliberately selected tails/bands.
        uniform = uniform[:256]
        global_indices = []
        for case in uniform:
            point = data['points'][case]
            doc = ds.document(int(point['document']))
            source = data['documents'][point['document']]
            actual = ds.token_ids[doc['token_start']:doc['token_start']+doc['token_count']]
            if not np.array_equal(actual, np.asarray(source['token_ids'], dtype=actual.dtype)):
                raise ValueError('Saved document token IDs differ from the canonical corpus')
            global_indices.append(doc['token_start'] + point['position'])
        values = sampled_products(ds, ds.neurons, global_indices)
        target_row = next(i for i, n in enumerate(ds.neurons) if n['id'] == target)
        expected = np.asarray([data['points'][p]['values'][2] for p in uniform], dtype=np.float32)
        if not np.array_equal(values[target_row], expected):
            raise ValueError('Target activations differ at the aligned saved positions')
        centered = values.astype(np.float64) - values.mean(axis=1, keepdims=True, dtype=np.float64)
        norms = np.linalg.norm(centered, axis=1)
        denom = norms * norms[target_row]
        correlation = np.divide(centered @ centered[target_row], denom,
                                out=np.zeros(len(values)), where=denom > 0)
        correlation = np.clip(correlation, -1., 1.)
        eligible = [i for i in range(len(values)) if i != target_row and norms[i] > 0 and np.isfinite(correlation[i])]
        positive = sorted([i for i in eligible if correlation[i] > 0],
                          key=lambda i: (-correlation[i], ds.neurons[i]['id']))[:6]
        negative = sorted([i for i in eligible if correlation[i] < 0],
                          key=lambda i: (correlation[i], ds.neurons[i]['id']))[:6]
        chosen = [target_row] + positive + negative
        units = [{'layer':ds.neurons[i]['layer'], 'neuron':ds.neurons[i]['neuron'],
                  'correlation':float(correlation[i]), 'sample_n':len(uniform),
                  'label':'this neuron' if i == target_row else 'sample r=%+.3f' % correlation[i]}
                 for i in chosen]
        selected = []
        used_docs = set()
        def add(case):
            case = str(case)
            doc = data['points'][case]['document']
            if doc not in used_docs:
                selected.append(case)
                used_docs.add(doc)
        for field in ['top_documents', 'bottom_documents']:
            for case in data['examples'][field][:4]:
                add(case)
        for band in data['bands']:
            if len(selected) >= 12:
                break
            if band['points']:
                add(band['points'][0])
        measurements = []
        for case in selected:
            point = data['points'][case]
            source = data['documents'][point['document']]
            pos = point['position']
            lo, hi = max(0, pos-64), min(len(source['tokens']), pos+11)
            readouts = []
            for index in chosen:
                saved = ds.document_activations(int(point['document']), ds.neurons[index]['id'])
                if not np.array_equal(saved['token_ids'], np.asarray(source['token_ids'])):
                    raise ValueError('Joint readouts use different input token IDs')
                unit = ds.neurons[index]
                readouts.append({'layer':unit['layer'], 'neuron':unit['neuron'],
                                 'values':saved['activations'][lo:hi, 2].tolist(),
                                 'anchor_channels':saved['activations'][pos].tolist()})
            measurements.append({'case_id':case, 'source_document':point['document'],
                'source_start':lo, 'source_position':pos, 'anchor_index':pos-lo,
                'prompt':''.join(source['tokens'][lo:hi]), 'tokens':source['tokens'][lo:hi],
                'observed_prefix':''.join(source['tokens'][:pos+1]),
                'following_prediction_target':''.join(source['tokens'][pos+1:pos+11]),
                'input_token_ids':source['token_ids'][lo:hi], 'readouts':readouts})
        family = {'family':units, 'measurements':measurements,
                  'method':'Saved native-product coactivation. Neighbours selected by signed Pearson correlation on uniform saved token positions among the 1000 recorded neurons.',
                  'selection_n':len(uniform), 'candidate_neurons':len(ds.neurons),
                  'source_kind':'saved_coactivation', 'upstream':[], 'downstream':[]}
        write(folder/'sections.json', {'family':family})
        write(folder/'correlations.json', {
            'uniform_case_ids':uniform, 'canonical_token_indices':global_indices,
            'selection':'six highest positive and six lowest negative Pearson correlations; target excluded',
            'candidates':[{'id':n['id'], 'correlation':float(correlation[i]), 'varying':bool(norms[i] > 0)}
                          for i, n in enumerate(ds.neurons)],
            'scope':'Sample correlations across recorded neurons; no causal direction is measured.'})
        result = {'tool':'paper_sections','status':'complete','sections':['family'],
                  'uniform_tokens':len(uniform),'candidate_neurons':len(ds.neurons),
                  'joint_documents':len(measurements),'units':units,'new_model_forwards':0,
                  'llm_requests':0,'finished_at':now()}
    except Exception as exc:
        result = {'tool':'paper_sections','status':'incomplete','sections':[],
                  'error_type':type(exc).__name__,'error':str(exc),'new_model_forwards':0,
                  'llm_requests':0,'finished_at':now()}
        write(folder/'sections.json', {})
    finally:
        ds.close()
    write(folder/'tool-request.json', {'id':'saved-analysis','title':'Saved coactivation',
          'report':session.config['report'], 'output':str(folder)})
    write(folder/'result.json', result)
    return result


def semantic_clusters(spectrum):
    groups = {g['id']:g for g in spectrum.get('groups', [])}
    points = collections.defaultdict(list)
    for point in spectrum.get('points', []):
        points[point['group']].append(point)
    total = sum(len(v) for v in points.values()) or 1
    clusters = []
    for identity, selected in points.items():
        group = groups.get(identity, {'name':identity,'description':'','pole':'either'})
        ordered = sorted(selected, key=lambda p:(-abs(p['activation']),p['id']))
        clusters.append({'id':identity, 'name':group['name'], 'story':group.get('description',''),
            'pole':group.get('pole','either'), 'share':len(selected)/total,
            'n_points':len(selected), 'member_groups':[identity], 'member_routes':[],
            'key_units':[], 'examples':ordered[:3],
            'typical_activation':float(np.median([p['activation'] for p in selected]))})
    return sorted(clusters, key=lambda x:(-x['n_points'],x['id']))


def draft_tests(session, arguments):
    from blind_tests import call
    folder = session.root/'test-proposals'
    if (folder/'result.json').exists():
        return read(folder/'result.json')
    folder.mkdir()
    claim = session.prior['interpretation']['claim']
    # A separate writer sees the initial explanation, without activation samples or raw measurements.
    brief = {'title':claim.get('title'), 'description':claim.get('lede'),
             'tests_to_distinguish':arguments.get('contrasts', [])}
    write(folder/'frozen-brief.json', brief)
    count = int(arguments.get('candidates_per_pole',12))
    messages = [
        {'role':'system','content':'Design text tests for a neuron explanation. Return JSON with positive and negative arrays. Each item has text, rationale, and competing_prediction. Write diverse matched contrasts that could distinguish the explanation from alternatives. You are proposing unmeasured inputs.'},
        {'role':'user','content':json.dumps({'explanation':brief,'texts_per_pole':count},ensure_ascii=False)}]
    try:
        answer = call(messages, folder/'writer', 'test-proposal-writer', max_tokens=393216)
        candidates = []
        for pole in ['positive','negative']:
            items = answer.get(pole, [])
            if isinstance(items, str):
                items = [items]
            for item in items[:count]:
                if isinstance(item,str):
                    item = {'text':item}
                candidates.append({**item,'pole':pole})
        plan = {'candidates':candidates,'brief':brief,'hypotheses':arguments.get('hypotheses', []),
                'status':'not_measured'}
        write(folder/'sections.json', {'tests':{'plan':plan}})
        result = {'tool':'paper_sections','status':'complete','sections':['tests'],
                  'candidate_count':len(candidates),'new_model_forwards':0,
                  'note':'Proposed texts only; no activation or blind-test score is available.'}
    except Exception as exc:
        write(folder/'sections.json', {})
        result = {'tool':'paper_sections','status':'incomplete','sections':[],
                  'error_type':type(exc).__name__,'error':str(exc),'new_model_forwards':0}
    write(folder/'tool-request.json', {'id':'test-proposals','output':str(folder),'title':'Unmeasured text-test proposals'})
    write(folder/'result.json', result)
    session.jobs.append({'tool':'paper_sections','request':read(folder/'tool-request.json')})
    write(session.root/'experiment-index.json',session.jobs)
    return result
