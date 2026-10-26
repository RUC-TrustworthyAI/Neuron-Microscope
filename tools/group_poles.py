"""Annotate saved semantic groups with explicit response poles; no model forwards."""
import collections
import copy
import json
from pathlib import Path
from common import write

POLES = ('positive', 'negative', 'either')
SYSTEM = """Read the saved activation evidence and assign a response pole to each existing semantic group.
positive: the group's recurring semantic pattern is associated with a positive neuron product.
negative: the group's recurring semantic pattern is associated with a negative neuron product.
either: the group is mixed, background, unclear, or has insufficient evidence for a directional response.
A group can contain individual activations of the opposite sign. Judge the semantic pattern, the sizes of the responses, and the counterexamples together; do not classify by the median or a sign-count vote. These labels describe neuron responses, not sentiment.
The observed prefix ends at the measured anchor. Following text is a prediction target and was not yet observed at that position.
Keep the existing group definitions and sample assignments. Return JSON with this structure:
{"groups":[{"id":"existing group ID","pole":"positive|negative|either","reason":"brief evidence-based reason","supporting_case_ids":[],"counterexample_case_ids":[]}]}"""


def packet(spectrum):
    """Include every saved group member, with its measured anchor and saved context."""
    members = collections.defaultdict(list)
    for point in spectrum.get('points', []):
        trace = point.get('token_activations') or []
        anchor = point.get('peak_index')
        if trace and isinstance(anchor, int):
            context = {'anchor_token':trace[anchor]['token'],
                       'observed_prefix':''.join(t['token'] for t in trace[:anchor+1]),
                       'following_prediction_target':''.join(t['token'] for t in trace[anchor+1:]),
                       'cut_left':point.get('cut_left', False)}
        else:
            context = {'saved_text':point.get('text', ''), 'anchor_index':anchor}
        members[point['group']].append({
            'case_id':str(point['id']), 'product':point['activation'], **context})
    groups = []
    for group in spectrum.get('groups', []):
        samples = members[group['id']]
        values = [p['product'] for p in samples]
        groups.append({
            'id':group['id'], 'name':group['name'],
            'description':group.get('description', group.get('definition', '')),
            'sample_summary':{'n':len(samples), 'positive':sum(v>0 for v in values),
                              'negative':sum(v<0 for v in values), 'zero':sum(v==0 for v in values),
                              'min':min(values, default=None), 'max':max(values, default=None)},
            'samples':samples})
    return {'activation':'product', 'scope':spectrum.get('sampling'),
            'task':'Assign one explicit response pole to every listed semantic group from the saved evidence.',
            'groups':groups}


def annotate(spectrum, folder, llm=None):
    if llm is None:
        from blind_tests import call
        llm = call
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    evidence = packet(spectrum)
    write(folder/'evidence.json', evidence)
    messages = [{'role':'system', 'content':SYSTEM},
                {'role':'user', 'content':json.dumps(evidence, ensure_ascii=False)}]
    answer = llm(messages, folder/'call', 'semantic-group-poles', max_tokens=393216)
    labels = {str(row.get('id')):row for row in answer.get('groups', [])
              if row.get('pole') in POLES}
    result = copy.deepcopy(spectrum)
    for group in result.get('groups', []):
        if group['id'] in labels:
            group['pole'] = labels[group['id']]['pole']
    write(folder/'spectrum.json', result)
    write(folder/'result.json', {
        'status':'complete' if all(g.get('pole') in POLES for g in result.get('groups', [])) else 'partial',
        'groups':[{'id':g['id'], 'pole':g.get('pole')} for g in result.get('groups', [])],
        'llm_requests':1, 'new_model_forwards':0})
    return result


def cluster_poles(clusters, spectrum):
    """One-to-one semantic clusters inherit their group's explicit annotation."""
    poles = {g['id']:g['pole'] for g in spectrum.get('groups', []) if g.get('pole') in POLES}
    result = copy.deepcopy(clusters)
    for cluster in result:
        members = cluster.get('member_groups', [cluster['id']])
        if len(members) == 1 and members[0] in poles:
            cluster['pole'] = poles[members[0]]
    return result
