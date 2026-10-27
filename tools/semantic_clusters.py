"""Agent-defined semantic clusters over measured natural-atlas members.

Raw NMF routes stay unchanged. Membership and prose come from annotations;
counts, mass, upstream contributions and examples come from saved measurements.
"""
import collections
import copy
from pathlib import Path
from common import read
from measured_response import describe_group, semantic_definition


def catalog(folders):
    latest = None
    for directory in folders:
        folder = Path(directory)
        if (folder/'route-factors.json').is_file() and (folder/'atlas-raw.json').is_file():
            latest = folder
    if latest is None:
        return {'routes': [], 'members': {}, 'route_weights': {}, 'source': None}
    factors = read(latest/'route-factors.json')
    routes = factors.get('routes', [])
    membership = {identity: route['id'] for route in routes for identity in route.get('graph_ids', [])}
    members = {row['id']: row for row in read(latest/'atlas-raw.json')
               if row.get('status', 'complete') == 'complete' and row.get('id') in membership}
    weights = {}
    for matrix in factors.get('routes_matrix', {}).values():
        for identity, values in zip(matrix['graph_ids'], matrix['route_weights']):
            weights[identity] = dict(zip(matrix['route_ids'], values))
    return {'routes': routes, 'members': members, 'membership': membership,
            'route_weights': weights, 'source': str(latest)}


def evidence(folders, arguments, prior=None):
    data = catalog(folders)
    selected = set(map(str, arguments.get('ids', [])))
    rows = list(data['members'].values())
    if arguments.get('group'):
        rows = [row for row in rows if data['membership'][row['id']] == arguments['group']]
    if selected:
        rows = [row for row in rows if row['id'] in selected]
        items = [{**row, 'graph_id': row['id'],
                  'route_id': data['membership'][row['id']],
                  'route_weights': data['route_weights'].get(row['id'], {})} for row in rows]
    else:
        items = []
        for row in rows:
            text = row.get('observed_prefix', row['text'])
            preview = text if len(text) <= 400 else text[:160] + '\n…\n' + text[-240:]
            items.append({
                'graph_id': row['id'], 'route_id': data['membership'][row['id']],
                'case_id': row.get('case_id'), 'semantic_group': row.get('semantic_group'),
                'selected': row['selected'], 'context_preview': preview,
                'full_context_available': True,
                'upstream': sorted(row['upstream'], key=lambda u: -abs(u['weight']))[:8],
                'route_weights': data['route_weights'].get(row['id'], {})
            })
    return {
        'section': 'routes', 'n_members': len(data['members']),
        'scope': 'All natural training members of the latest measured routes; holdout cases and written probes stay separate.',
        'routes': [{k: r[k] for k in ('id', 'pole', 'n_graphs', 'graph_ids', 'upstream', 'share') if k in r}
                   for r in data['routes']],
        'members': items,
        'semantic_clusters': (prior or {}).get('cluster_definitions', []),
        'read_full_member': 'Use --section routes --ids GRAPH_ID to read the complete observed prefix, native trace and upstream vector.',
    }


def apply(bundle, definitions, folders):
    from paper_sections import exemplar
    data = catalog(folders)
    if not data['routes']:
        return
    members = data['members']
    definitions = [semantic_definition(d) for d in definitions]
    used = set()
    clusters = []
    total_mass = sum(sum(abs(u['weight']) for u in row['upstream']) for row in members.values()) or 1.
    spectrum = bundle.get('spectrum') or {}
    groups = {str(p['id']): p['group'] for p in spectrum.get('points', [])}

    def materialize(definition, identities):
        rows = [members[identity] for identity in identities]
        scores = collections.defaultdict(float)
        for row in rows:
            for unit in row['upstream']:
                scores[(unit['layer'], unit['neuron'])] += unit['weight']
        norm = sum(abs(value) for value in scores.values()) or 1.
        units = [{'layer': key[0], 'neuron': key[1], 'weight': value/norm}
                 for key, value in sorted(scores.items(), key=lambda item: -abs(item[1]))[:8]]
        example_ids = list(dict.fromkeys(str(x) for x in definition.get('example_graph_ids', [])))
        examples = [{**exemplar(members[identity]), 'graph_id': identity}
                    for identity in example_ids if identity in identities]
        member_groups = list(dict.fromkeys(
            groups.get(str(row.get('case_id')), row.get('semantic_group'))
            for row in rows))
        return describe_group({
            'id': definition['id'], 'name': definition['name'],
            'story': definition.get('story', ''),
            'graph_ids': identities, 'example_graph_ids': [e['graph_id'] for e in examples],
            'member_routes': list(dict.fromkeys(data['membership'][identity] for identity in identities)),
            'member_groups': [g for g in member_groups if g],
            'n_graphs': len(rows),
            'share': sum(sum(abs(u['weight']) for u in row['upstream']) for row in rows)/total_mass,
            'key_units': units, 'examples': examples,
        }, [row['selected']['product'] for row in rows])

    for definition in definitions:
        # Each measured member is counted once; no measurement is removed.
        identities = [str(x) for x in definition.get('graph_ids', [])
                      if str(x) in members and str(x) not in used]
        identities = list(dict.fromkeys(identities))
        used.update(identities)
        if identities:
            clusters.append(materialize(definition, identities))
    remaining = [identity for identity in members if identity not in used]
    if remaining:
        clusters.append(materialize({
            'id': 'unassigned-measured-members',
            'name': 'Unassigned measured contexts', 'pole': 'either', 'story': '',
            'example_graph_ids': remaining[:3],
        }, remaining))
    bundle['clusters'] = clusters
    bundle['cluster_definitions'] = definitions
    bundle['cluster_source'] = 'agent_semantic_routes'
