"""Assemble saved activation evidence for on-demand agent access."""
import collections
import datetime as dt
import json
import os
from pathlib import Path
from tokenizers import Tokenizer
import unicodedata

def read(p):
    return json.loads(Path(p).read_text())

def save(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, allow_nan=False, indent=2) + '\n')

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def saved_peak_census(data, cases, tokenizer, pole):
    if pole == 'absolute':
        points = list(dict.fromkeys(data['examples']['top_tokens'] + data['examples']['bottom_tokens']))
        points.sort(key=lambda p: (-abs(cases[str(p)]['anchor']['product']), int(p)))
    else:
        key = 'top_tokens' if pole == 'positive' else 'bottom_tokens'
        points = [p for p in data['examples'][key] if (cases[str(p)]['anchor']['product'] > 0 if pole == 'positive' else cases[str(p)]['anchor']['product'] < 0)]
    classes = read(Path(__file__).with_name('peak-token-classes.json'))
    words = set(classes['function_words'])
    peaks = []
    for p in points[:30]:
        case = cases[str(p)]
        a = case['anchor']
        token = a['token']
        text = token.strip()
        before = case['observed_prefix'][:-len(token)] if token and case['observed_prefix'].endswith(token) else ''
        after = case['prediction_target']
        if not text:
            kind = 'space / newline'
        elif '�' in token:
            kind = 'token fragment'
        elif text.isdecimal():
            kind = 'digit'
        elif all((unicodedata.category(c)[0] in 'PS' for c in text)):
            kind = 'punctuation'
        elif text.casefold() in words and (token[-1].isspace() or not after or (not after[0].isalnum())) and (token[0].isspace() or not before or (not before[-1].isalnum()) or (text in ("'s", '’s', "n't"))):
            kind = 'function word'
        elif text[0].isupper():
            kind = 'capitalised word'
        else:
            kind = 'other word'
        peaks.append(dict(global_token=int(p), token=token, kind=kind, product=a['product']))
    return dict(peaks=peaks, selection_unit='token', ranking={'positive': 'native_product_desc', 'negative': 'native_product_asc', 'absolute': 'absolute_native_product_desc'}[pole] + '_then_global_token_asc', classification=classes, derived_from_saved_extrema=True, source='Top30 from saved exact token extrema; absolute Top30 is selected from the union of signed Top30 extrema. No new activation scan.')

def prepare(report, out, tokenizer=None):
    d = read(report / 'report.json')
    if tokenizer is None:
        from settings import load, paths
        tokenizer=paths(load())['tokenizer']
    tok = Tokenizer.from_file(str(tokenizer))
    groups = collections.defaultdict(list)
    for (group, ids) in d['examples'].items():
        for p in ids:
            groups[str(p)].append(group)
    for band in d['bands']:
        for p in band['points']:
            groups[str(p)].append(f"p{band['low']}-p{band['high']}")
    for p in d['scatter']:
        if str(p) in groups:
            groups[str(p)].append('scatter')
    (cases, by_id) = ([], {})
    for (p, labels) in groups.items():
        a = d['points'][p]
        doc = d['documents'][a['document']]
        pos = a['position']
        ids = doc['token_ids']
        case = dict(id=p, document=a['document'], groups=labels, anchor=dict(position=pos, token_id=ids[pos], token=doc['tokens'][pos], gate=a['values'][0], value=a['values'][1], product=a['values'][2]), observed_prefix=tok.decode(ids[:pos + 1], skip_special_tokens=False), prediction_target=tok.decode(ids[pos + 1:pos + 11], skip_special_tokens=False), trace=[[j - pos, doc['tokens'][j], doc['gate'][j], doc['value'][j], doc['product'][j]] for j in range(max(0, pos - 24), min(len(ids), pos + 11))])
        (prefix, anchor_text) = (case['observed_prefix'], case['anchor']['token'])
        if anchor_text and prefix.endswith(anchor_text):
            quote = prefix[:-len(anchor_text)][-200:] + '⟦' + anchor_text + '⟧'
        else:
            quote = prefix[-200:] + '\nAnchor token: ' + json.dumps(anchor_text, ensure_ascii=False)
        case['evidence'] = dict(text=quote + '\nFollowing prediction target: ' + case['prediction_target'], activation=case['anchor']['product'])
        cases.append(case)
        by_id[p] = case
    bands = [{k: v for (k, v) in b.items() if k != 'points'} for b in d['bands']]
    tables = []
    for t in read(report / 'token-rankings.json'):
        v = t['view']
        rows = []
        for row in t['rows'][:20]:
            (tid, text, count, mean) = row[:4]
            hits = row[v['column']]
            rows.append([tid, text, count, mean, hits, hits / count])
        extreme = d['statistics']['product']['max' if v['mode'] == 'positive' else 'min']
        tables.append(dict(pole=v['mode'], extreme_fraction=v['fraction'], reference_extreme=extreme, threshold=v['threshold'], comparison='>' if v['mode'] == 'positive' else '<', rows=rows))
    census = {}
    for (name, filename) in [('positive', 'peak-census.json'), ('negative', 'peak-census-negative.json'), ('absolute_appendix', 'peak-census-absolute.json')]:
        original = read(report / filename) if (report / filename).exists() else saved_peak_census(d, by_id, tok, 'absolute' if name == 'absolute_appendix' else name)
        selected = original['peaks'][:30]
        rows = []
        for x in selected:
            p = str(x['global_token'])
            rows.append(dict(case_id=p, token=x['token'], kind=x['kind'], product=x['product']))
        census[name] = dict(n=len(rows), selection_unit=original['selection_unit'], ranking=original['ranking'], distinct_documents=len({by_id[x['case_id']]['document'] for x in rows}), tokens=dict(collections.Counter((x['token'] for x in rows))), kinds=dict(collections.Counter((x['kind'] for x in rows))), classification=original['classification'], peaks=rows)
        if original.get('derived_from_saved_extrema'):
            census[name]['source'] = original['source']
    effects = read(report / 'effects.json') if (report / 'effects.json').exists() else {'logit_effect': None, 'ablation': None}
    for row in (effects.get('ablation') or {}).get('rows', []):
        p = str(row['case_id'])
        row['context_case_id'] = p
        row.pop('text', None)
    evidence = dict(distribution_json=dict(channels=d['statistics'], bands=bands, methods=d['methods']), contexts_json=dict(trace_columns=['offset', 'token', 'gate', 'value', 'product'], cases=cases), token_selectivity_json=dict(threshold_definition='threshold = extreme_fraction * reference_extreme; reference_extreme is the corpus product maximum for positive tables and minimum for negative tables. This is an activation-magnitude ratio, not a percentile or a fraction of tokens.', hit_fraction_definition='fraction = hits / occurrences for this token type; it is not extreme_fraction.', mean_product_definition='mean_product is the mean over all occurrences of this token type, not an activation at one position.', minimum_occurrences=50, columns=['token_id', 'token', 'occurrences', 'mean_product', 'hits', 'fraction'], tables=tables), peak_census_json=census, gate_value_json=dict(sign_axes=['negative', 'zero', 'positive'], full_population_sign_counts=d['gate_value_sign_counts'], scatter_method=d['methods']['scatter'], scatter_columns=['gate', 'value', 'product'], scatter=[d['points'][str(p)]['values'] for p in d['scatter']]), output_effects_json=effects)
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'input.json', evidence)
    save(out / 'manifest.json', dict(prepared_at=now(),
        files=[dict(path=str(report / 'report.json'))],
        cases=len(cases), documents=len({c['document'] for c in cases})))
    return evidence
