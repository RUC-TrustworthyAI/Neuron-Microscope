"""Append follow-up Criticism to a report copy without changing other sections."""
import json
from pathlib import Path

from ..tools.report import BUNDLE, encode_bundle
from .state import write


def criticism_row(investigation, round_number):
    question = investigation['question']
    row = {'round': round_number, 'investigation_id': investigation['id'],
           'criticism_ids': question['criticism_ids'],
           'attack': f"Follow-up round {round_number}: {question['question']}",
           'prediction': f"Hypothesis ({question['hypothesis']}): {question['prediction_if_hypothesis']}\n"
                         f"Rival ({question['rival']}): {question['prediction_if_rival']}",
           'evidence': [], 'evidence_ids': []}
    if investigation['status'] == 'failed':
        return {**row, 'verdict': 'inconclusive', 'suggested_revision': '',
                'result': 'Investigation failed: ' + str(investigation['error'])}
    assessment = investigation['assessment']
    revision = assessment['suggested_revision']
    evidence = {entry['id']: entry for entry in investigation['evidence']['rows']}
    row['evidence_ids'] = assessment['evidence_ids']
    for identity in assessment['evidence_ids']:
        entry = evidence[identity]
        if 'trace' in entry:
            entry = {'text': entry['text'], 'activation': entry['selected']['product']}
        if 'text' in entry and 'activation' in entry:
            row['evidence'].append(entry)
    return {**row, 'verdict': assessment['verdict'], 'suggested_revision': revision,
            'result': assessment['result'] + (f'\nSuggested revision: {revision}' if revision else '') +
                      ('\nEvidence: ' + ', '.join(assessment['evidence_ids']) if assessment['evidence_ids'] else '')}


def render(source_html: str, rounds: list, output: Path) -> Path:
    from ..tools.report_layout import refresh_criticism

    matches = list(BUNDLE.finditer(source_html))
    if len(matches) != 1:
        raise ValueError('Expected exactly one source report bundle')
    bundle = json.loads(matches[0][2])
    bundle['criticism'] = [*bundle.get('criticism', []), *(
        criticism_row(investigation, record['round'])
        for record in rounds for investigation in record['investigations'])]
    page = BUNDLE.sub(lambda match: match[1] + encode_bundle(bundle) + match[3], source_html, count=1)
    page = refresh_criticism(page)
    report = Path(output) / 'report'
    write(report / 'paper-data.json', {'bundle': bundle})
    temporary = report / 'index.html.tmp'
    temporary.write_text(page, encoding='utf-8')
    temporary.replace(report / 'index.html')
    return report
