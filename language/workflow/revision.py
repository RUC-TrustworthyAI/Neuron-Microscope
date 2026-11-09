"""Analyze experiments while preserving the frozen explanation and predictions."""
from . import contracts, prompts
from .evidence import prediction_feedback


def analyze(client, initial, predictions, packet, evidence, timeline):
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
    return {**analysis, **{key: initial[key] for key in ('claim', 'response_rules')}}

def review_report(client, revision, predictions, packet, timeline, followups=None):
    timeline.status('independent-review')
    review_packet = {'claim': revision['claim'], 'response_rules': revision['response_rules'],
         'criticism': revision['criticism'], 'clusters': revision['clusters'],
         'search_progress': packet['search_progress'], 'controls': packet['controls'],
         'blind_tests': packet['blind_tests'], 'prediction_feedback': prediction_feedback(predictions)}
    if followups and followups['status'] != 'disabled':
        from .criticism_report import history_for_planning
        review_packet['followups'] = {'status': followups['status'],
            'stop_reason': followups['stop_reason'], 'rounds': history_for_planning(followups['rounds'])}
    return client.request('final-review', prompts.REVIEW, review_packet, contracts.REVIEW, 4)
