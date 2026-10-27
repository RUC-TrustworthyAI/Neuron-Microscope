"""Connect saved evidence and completed measurements to report fields."""
import copy
from pathlib import Path
from common import read


def peak_fields(evidence):
    fields = {}
    names = {'positive': 'peak_census', 'negative': 'peak_census_negative',
             'absolute_appendix': 'peak_census_absolute'}
    for name, field in names.items():
        value = (evidence.get('peak_census_json') or {}).get(name)
        if not value:
            continue
        value = copy.deepcopy(value)
        for key, label in [('kinds', 'kind'), ('tokens', 'token')]:
            if isinstance(value.get(key), dict):
                value[key] = [{label: k, 'n': n} for k, n in value[key].items()]
        fields[field] = value
    return fields


def default_anatomy(bundle, data):
    """Keep the saved strip and recover its identity; otherwise use the top document."""
    claim = (bundle.get('claim') or {}).get('one_paragraph')
    if not claim or claim == '—':
        return None
    anatomy = copy.deepcopy(bundle.get('anatomy') or {})
    strip = anatomy.get('strip') or {}
    if strip.get('case_id'):
        return None
    points, documents = data.get('points', {}), data.get('documents', {})
    preferred = [str(x) for x in data.get('examples', {}).get('top_documents', [])]
    candidates = list(dict.fromkeys(preferred + list(points)))
    pairs = [('tokens', 'tokens'), ('gate', 'gate'), ('up', 'value'), ('act', 'product')]
    if strip.get('tokens'):
        n = len(strip['tokens'])
        for case_id in candidates:
            point = points[case_id]
            doc = documents[str(point['document'])]
            pos = point['position']
            for start in range(max(0, pos - n + 1), min(pos, len(doc['tokens']) - n) + 1):
                if all(strip.get(a) == doc[b][start:start+n] for a, b in pairs):
                    anatomy['strip'] = {**strip, 'case_id': case_id, 'source_start': start,
                                        'anchor_index': pos-start}
                    return anatomy
    if preferred:
        case_id = preferred[0]
        point = points[case_id]
        doc = documents[str(point['document'])]
        pos = point['position']
        start, stop = max(0, pos-24), min(len(doc['tokens']), pos+11)
        anatomy['strip'] = {a: doc[b][start:stop] for a, b in pairs}
        anatomy['strip'].update(text=''.join(doc['tokens'][start:stop]),
                               case_id=case_id, source_start=start, anchor_index=pos-start)
        return anatomy
    return None


def sweep_search(folders, existing, figures):
    """Expose saved scored conditions without changing the original scoring rule."""
    if (existing or {}).get('regions'):
        return None
    titles = {f['id']: f.get('title') for f in figures}
    regions, seen = [], set()
    for folder in folders:
        folder = Path(folder)
        request = read(folder/'tool-request.json')
        result = read(folder/'result.json')
        identity = request.get('id', folder.name)
        if identity in seen:
            continue
        seen.add(identity)
        conditions = [c for c in result.get('conditions', [])
                      if c.get('status') == 'complete' and c.get('selected') and c.get('trace')]
        if not conditions:
            continue
        hi = max(conditions, key=lambda c: c['selected']['product'])
        lo = min(conditions, key=lambda c: c['selected']['product'])
        frontier = []
        for condition in [hi] + ([] if lo is hi else [lo]):
            selected, trace = condition['selected'], condition['trace']
            peak_index = next(i for i, t in enumerate(trace) if t['position'] == selected['position'])
            frontier.append({
                'text': condition['text'], 'activation': selected['product'],
                'token_activations': [{'token': t['token'], 'activation': t['product']} for t in trace],
                'peak_index': peak_index, 'cut_left': False, 'cut_right': False,
                'condition': condition['condition'], 'scored_positions': condition['scored_positions']})
        regions.append({
            'id': identity, 'name': titles.get(identity) or request.get('title') or identity,
            'source_kind': 'saved_replacement_sweep', 'score': result.get('score', request['score']),
            'n': len(conditions), 'expected_conditions': result.get('expected_conditions'),
            'best': hi['selected']['product'], 'maximum': hi['selected']['product'],
            'minimum': lo['selected']['product'], 'frontier': frontier})
    if not regions:
        return None
    return {**(existing or {}), 'regions': regions, 'n_regions': len(regions),
            'n_texts': sum(r['n'] for r in regions), 'source_kind': 'saved_replacement_sweeps',
            'best_positive': max((r['maximum'] for r in regions if r['maximum'] > 0), default=None),
            'best_negative': min((r['minimum'] for r in regions if r['minimum'] < 0), default=None)}

def render(source):
    """Update the shared renderer, not individual generated pages."""
    import re
    from exemplar_display import render as render_exemplars
    source = render_exemplars(source)
    if "B.cluster_source==='saved_semantic_groups'?" not in source:
        source=source.replace("'Share of attribution mass per cluster.'",
            "(B.cluster_source==='saved_semantic_groups'?'Share of saved sample positions per semantic group.':'Share of attribution mass per cluster.')")
    if 'function peakCensusFigure(' not in source:
        source = source.replace('// Header: eyebrow, title, short lede, headline figure, quiet meta row.',
                                "// Saved signed peak census; absolute ranking stays in the appendix.\nfunction peakCensusFigure(PC, label) {\n  if (!has(PC) || (!has(PC.kinds) && !has(PC.tokens))) return null;\n  const kinds = [...(PC.kinds || [])].sort((a,b)=>(b.n||0)-(a.n||0)), km = Math.max(...kinds.map(k=>k.n||0),1);\n  const bars = $('div',{class:'kbars'},kinds.slice(0,7).map(k=>$('div',{class:'kb'},$('span',{class:'kl'},k.kind),$('span',{class:'kt'},$('i',{style:`width:${100*(k.n||0)/km}%`})),$('span',{class:'kn'},k.n))));\n  const tokens = [...(PC.tokens || [])].sort((a,b)=>(b.n||0)-(a.n||0)).slice(0,14);\n  const chips = $('div',{class:'chips',style:'margin:0'},tokens.map(t=>$('span',{class:'tokchip'},JSON.stringify(t.token),' ',$('b',{style:'color:var(--ink);font-weight:500'},t.n))));\n  const node = fig($('div',null,$('div',{class:'sub'},label),$('div',{class:'census'},$('div',null,$('div',{class:'sub'},'Token kind at the peak'),bars),$('div',null,$('div',{class:'sub'},'Most common peak tokens'),chips))),`${PC.n} token positions · ${PC.distinct_documents} distinct documents.`);\n  if (PC.note) node.title=PC.note;\n  return node;\n}" + '\n\n// Header: eyebrow, title, short lede, headline figure, quiet meta row.')
        source = re.sub(r'  let census = null; const PC = B\.peak_census;.*?  if \(!parts\.length && !census\) return; sec\(\x27claim\x27, \x27The claim in detail\x27, null, parts\.length \? \$\(\x27div\x27, \{ class: \x27detail\x27 \}, parts\) : null, census\);',
            "  const census = peakCensusFigure(B.peak_census, 'Positive Top' + (B.peak_census?.n || ''));\n"
            "  const negative = peakCensusFigure(B.peak_census_negative, 'Negative Top' + (B.peak_census_negative?.n || ''));\n"
            "  if (!parts.length && !census && !negative) return; sec('claim', 'The claim in detail', null, parts.length ? $('div', {class:'detail'}, parts) : null, census, negative);",
            source, count=1, flags=re.S)
        source = source.replace("(() => { const A = B.appendix || {}; const parts = [];",
            "(() => { const A = B.appendix || {}; const parts = [];\n"
            "  const absolute = peakCensusFigure(B.peak_census_absolute, 'Absolute Top' + (B.peak_census_absolute?.n || ''));\n"
            "  if(absolute)parts.push($('details',null,$('summary',null,'Absolute peak census'),absolute));")
    # One publication panel pools both saved tails; the source fields remain separate.
    source=re.sub(r'(?:function combinedPeakCensus\(.*?|function peakCensusFigure\(.*?)' +
                  r'(?=\n\n// Header:)',
                  lambda m:Path(__file__).with_name('peak_census_display.js').read_text().rstrip(),
                  source,count=1,flags=re.S)
    source=re.sub(r"  const census = peakCensusFigure\(B\.peak_census,.*?null, census, negative\);",
                  "  const census = peakCensusFigure(combinedPeakCensus(B.peak_census,B.peak_census_negative));\n"
                  "  if (!parts.length && !census) return; sec('claim', 'The claim in detail', null, parts.length ? $('div', {class:'detail'}, parts) : null, census);",
                  source,count=1,flags=re.S)
    source = source.replace("return $('div', { class: 'region' }, $('h4', null, r.name), $('div', { class: 'best' }, r.n != null ? `${r.n} probes · best ` : 'best ', tile(r.best, { signed: true, scale: G })), (r.frontier || []).slice(0, 3).map(e => exRow(e, s, (e.activation || 0) < 0, G)));", "const best = r.source_kind === 'saved_replacement_sweep'\n      ? $('div',{class:'best'},`${r.n} scored conditions · max `,tile(r.maximum,{signed:true,scale:G}),' · min ',tile(r.minimum,{signed:true,scale:G}))\n      : $('div',{class:'best'},r.n != null ? `${r.n} probes · best ` : 'best ',tile(r.best,{signed:true,scale:G}));\n    return $('div',{class:'region'},$('h4',null,r.name),best,(r.frontier||[]).slice(0,3).map(e=>exRow(e,s,(e.activation||0)<0,G)));")
    source = source.replace("[P(`Written probes, region by region. Each region is a hypothesis about what drives the unit; the frontier is the best text found there. ${S.n_regions ? S.n_regions + ' regions, ' : ''}${S.n_texts ? S.n_texts.toLocaleString() + ' texts scored. ' : ''}A region whose best value stays far below the corpus maximum is not where the unit lives.`)]", "[P(S.source_kind==='saved_replacement_sweeps' ? `${S.n_regions} saved replacement sweeps · ${S.n_texts} scored conditions. Values retain each sweep’s original scoring rule.` : `Written probes, region by region. Each region is a hypothesis about what drives the unit; the frontier is the best text found there. ${S.n_regions ? S.n_regions + ' regions, ' : ''}${S.n_texts ? S.n_texts.toLocaleString() + ' texts scored. ' : ''}A region whose best value stays far below the corpus maximum is not where the unit lives.`)]")
    source = source.replace("...S.regions.map(r => Math.abs(r.best || 0))",
                            "...S.regions.flatMap(r => [r.best, r.minimum].map(v => Math.abs(v || 0)))")
    return source
