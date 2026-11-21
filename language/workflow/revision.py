"""Analyze experiments while preserving the frozen explanation and predictions."""
import hashlib
import json

from . import contracts, prompts
from .evidence import prediction_feedback
from .state import read


def ablation_analysis_packet(section, evidence):
    rows = sorted(section['rows'], key=lambda row: -row['kl'])[:5]
    fields = ('kind', 'text', 'peak_token', 'live_product', 'kl', 'kl_controls', 'gains', 'losses')
    examples = [{**{field: row[field] for field in fields},
                 'evidence_id': row['source_id'] if row['source_id'] in evidence else None}
                for row in rows]
    return {'summary': section['summary'], 'examples': examples}


def resume_graph_ablation_packet(root, packet):
    """Honor a completed analysis request from before Effect was expanded."""
    folder = root / 'api/report-analysis'
    result = folder / 'result.json'
    if not result.is_file():
        return False
    fingerprint = read(result)['input_id']
    for path in sorted(folder.glob('attempt-*/request.json')):
        messages = read(path)['messages'][:2]
        if hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest() != fingerprint:
            continue
        prior = json.loads(messages[1]['content']).get('matched_ablations', {})
        if 'replaces_case_ids' in prior:
            packet['matched_ablations'] = prior
            return True
        break
    return False


def finalize(client, initial, predictions, packet, evidence, sample, contexts, root, timeline):
    valid_atlas = {row['id'] for row in packet['atlas']}

    def check_revision(output):
        for criticism in output['criticism']:
            unknown = set(criticism['evidence_ids']) - evidence.keys()
            if unknown:
                raise ValueError('Unknown case IDs in criticism.evidence_ids: ' + ', '.join(sorted(unknown)) +
                    '. Cite only the supplied ID of an individual measured case. Vocabulary token IDs '
                    'and token-selectivity table labels are not case IDs. Discuss aggregate statistics '
                    'in the result text without adding an evidence ID for them.')
        used = [identity for cluster in output['clusters'] for identity in cluster['graph_ids']]
        if set(used) - valid_atlas or len(used) != len(set(used)):
            raise ValueError('Semantic cluster member IDs must refer to measured atlas cases and occur once')
        long_labels = [f"{unit['id']}: {unit['label']} ({len(unit['label'].split())} words)"
                       for unit in output['units'] if len(unit['label'].split()) > 8]
        if long_labels:
            raise ValueError('Shorten only these unit labels to at most eight words each: ' + '; '.join(long_labels))

    timeline.status('report-analysis')
    analysis = client.request('report-analysis', prompts.REVISE,
        {**packet, 'current_explanation': initial, 'prediction_feedback': prediction_feedback(predictions)},
        contracts.REPORT_ANALYSIS, 4, check_revision)
    revision = {**analysis, **{key: initial[key] for key in ('claim', 'response_rules')}}

    timeline.status('independent-review')
    review = client.request('final-review', prompts.REVIEW,
        {'claim': revision['claim'], 'response_rules': revision['response_rules'],
         'criticism': revision['criticism'], 'clusters': revision['clusters'],
         'search_progress': packet['search_progress'], 'controls': packet['controls'],
         'blind_tests': packet['blind_tests'], 'prediction_feedback': prediction_feedback(predictions)}, contracts.REVIEW, 4)
    return revision, predictions, review
