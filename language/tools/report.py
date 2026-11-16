"""Merge measured artifacts into a new original-style report copy."""
import copy
import html
import json
import math
import re
from pathlib import Path

BUNDLE = re.compile(r'(<script\b(?=[^>]*\bid=["\x27]bundle["\x27])[^>]*>)(.*?)(</script>)', re.S)
EFFECT = re.compile(r'// 5b\. What it does:.*?(?=// 5c\. Where it sits:)', re.S)
TESTS = re.compile(r'// 7\. Tests\..*?(?=// 8\. Criticism[.:])', re.S)
ANATOMY_PAIR = re.compile(r'// Task 3 paired Anatomy renderer\..*?// End paired Anatomy renderer\.', re.S)
ANATOMY_INTRO = re.compile(r"(sec\(\{ id: 'anatomy', wide: true \}, 'Inside the neuron', \[P\(')([^']+)(')")
NEURON_LABEL_SCRIPTS = re.compile(
    r'<script id="neuron-labels" type="application/json">(?P<data>.*?)</script>'
    r'<style id="neuron-label-style">.*?</style>'
    r'<script id="neuron-label-render">.*?</script>', re.S)
NEURON_LABEL_HOOK = re.compile(r'/\* neuron-labels:start \*/.*?/\* neuron-labels:end \*/', re.S)
REPORT_SYNTHESIS = re.compile(
    r'<script id="report-synthesis" type="application/json">(?P<data>.*?)</script>'
    r'<script id="report-synthesis-render">.*?</script>', re.S)
REPORT_TEMPLATE = Path(__file__).resolve().parents[1] / 'analysis/report.html'


def _refresh_renderer(source, pattern, description):
    block = pattern.search(REPORT_TEMPLATE.read_text(encoding='utf-8'))
    if block is None or pattern.search(source) is None:
        raise ValueError(f'Expected {description} renderer in both the saved report and template')
    return pattern.sub(lambda _: block[0], source, count=1)


def refresh_effect(source):
    return _refresh_renderer(source, EFFECT, 'an Effect')


def refresh_tests(source):
    """Refresh only the Tests renderer in a saved report."""
    return _refresh_renderer(source, TESTS, 'a Tests')


def encode_bundle(bundle):
    return json.dumps(bundle, ensure_ascii=False, allow_nan=False, separators=(',', ':')).replace(
        '<', '\\u003c').replace('\u2028', '\\u2028').replace('\u2029', '\\u2029')


def _replace_bundle(source, bundle):
    match = BUNDLE.search(source)
    return source[:match.start(2)] + encode_bundle(bundle) + source[match.end(2):]


def neuron_labels_overlay(result, evidence):
    """Resolve qualitative labels to their cited discovery measurements."""
    labels, citations = {}, {}
    samples = evidence.get('samples') or {}
    for unit, label in (result.get('labels') or {}).items():
        by_id = {
            sample['id']: sample for sample in samples.get(unit, [])
            if sample.get('id') and sample.get('neuron_id') == unit
            and sample.get('split') == 'discovery'
            and type(sample.get('activation')) in (int, float)
            and math.isfinite(sample['activation'])
        }
        cited_ids = [key for key in dict.fromkeys(label.get('evidence_ids') or []) if key in by_id]
        labels[unit] = {field: label.get(field, '')
                        for field in ('label', 'description', 'status', 'positive', 'negative')}
        labels[unit]['evidence_ids'] = cited_ids
        citations.update({key: copy.deepcopy(by_id[key]) for key in cited_ids})
    return {'labels': labels, 'evidence': citations, 'scope': copy.deepcopy(evidence.get('scope') or {})}


def strip_neuron_labels(source):
    """Recover the exact HTML used as input to the label overlay."""
    scripts = NEURON_LABEL_SCRIPTS.search(source)
    if scripts is None:
        return source
    renderer, end_tag, tail = source[scripts.end():].partition('</script>')
    return source[:scripts.start()] + NEURON_LABEL_HOOK.sub('', renderer) + end_tag + tail


def update_neuron_labels(source, result, evidence):
    """Render functional labels as a reversible overlay on a saved report."""
    from report_layout import render_neuron_labels

    source = strip_neuron_labels(source)
    bundle, _ = source_bundle(source)
    overlay = neuron_labels_overlay(result, evidence)
    bundle['neuron_labels'] = overlay
    insertion = BUNDLE.search(source).end()
    renderer, end_tag, tail = source[insertion:].partition('</script>')
    scripts = (f'<script id="neuron-labels" type="application/json">{encode_bundle(overlay)}</script>'
               f'<style id="neuron-label-style">{NEURON_LABEL_STYLE}</style>'
               '<script id="neuron-label-render">'
               + Path(__file__).with_name('neuron_label_display.js').read_text(encoding='utf-8')
               + '</script>')
    page = source[:insertion] + scripts + render_neuron_labels(renderer) + end_tag + tail
    return page, bundle


NEURON_LABEL_STYLE = """
.neuron-label-control{cursor:pointer;text-decoration:underline;text-underline-offset:3px}
button.neuron-label-control{border:0;padding:0;color:inherit;background:none;font:inherit;text-align:left}
.chip.neuron-label-control{text-decoration:none}
.chip.neuron-label-control .lbl{text-decoration:underline;text-underline-offset:3px}
#neuron-label-evidence{box-sizing:border-box;width:min(680px,calc(100% - 32px));max-height:calc(100% - 48px);border:1px solid var(--rule);border-radius:9px;padding:20px;color:var(--ink);background:#fff}
#neuron-label-evidence::backdrop{background:#1d1d1f66}
#neuron-label-evidence h2{margin:8px 0;font-size:22px;line-height:1.2}
#neuron-label-evidence p{margin:8px 0}
#neuron-label-evidence .neuron-label-meta{font-size:12px;color:var(--ink3);overflow-wrap:anywhere}
#neuron-label-evidence .neuron-label-close{float:right}
#neuron-label-evidence li{margin-bottom:12px}
#neuron-label-evidence .neuron-label-context{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 var(--mono);padding:8px 10px;background:#f6f6f7}
#neuron-label-evidence mark{background:#ffe4bd;color:inherit;border-radius:2px;box-shadow:0 1px #a95020}
svg .neuron-label-control{cursor:pointer}
svg .neuron-label-control:focus-visible{outline:2px solid currentColor;outline-offset:3px}
"""


def refresh_anatomy_pair(source):
    """Insert the pair renderer without replacing saved Anatomy plots or layout adapters."""
    canonical = REPORT_TEMPLATE.read_text(encoding='utf-8')
    block = ANATOMY_PAIR.search(canonical)
    intro = ANATOMY_INTRO.search(canonical)
    if block is None or intro is None:
        raise ValueError('Missing paired Anatomy renderer in report template')
    source = ANATOMY_INTRO.sub(lambda match: match[1] + intro[2] + match[3], source, count=1)
    source = source.replace('have the gate open;', 'have positive raw gate;').replace(
        'Points in the shaded left band are off whatever up says.',
        'The shaded left band marks negative raw gate values.')
    source = source.replace("@media (max-width:1000px){.topnav a[href='#anatomy'],",
                            '@media (max-width:1000px){')
    if ANATOMY_PAIR.search(source):
        return ANATOMY_PAIR.sub(lambda _: block[0], source, count=1)
    legacy = '  if (has(A.headline)) figs.push('
    if source.count(legacy) != 1:
        raise ValueError('Expected one legacy Anatomy headline renderer')
    return source.replace(legacy, '  ' + block[0] + '\n'
                          '  if (!hasAnatomyPair && has(A.headline)) figs.push(', 1)


def with_neuron_labels(source, bundle):
    """Include the label overlay in the report data."""
    labels = NEURON_LABEL_SCRIPTS.search(source)
    if labels is None:
        return bundle
    return {**bundle, 'neuron_labels': json.loads(labels['data'])}


def strip_report_synthesis(source):
    """Recover the report that was used to generate its synthesis."""
    return REPORT_SYNTHESIS.sub('', source, count=1)


def update_report_synthesis(source, synthesis):
    """Display the measured synthesis as a reversible report overlay."""
    renderer = Path(__file__).with_name('report_synthesis.js').read_text(encoding='utf-8')
    scripts = (f'<script id="report-synthesis" type="application/json">{encode_bundle(synthesis)}</script>'
               f'<script id="report-synthesis-render">{renderer}</script>')
    head, closing, tail = strip_report_synthesis(source).partition('</body>')
    page = head + scripts + closing + tail
    return page, with_neuron_labels(page, source_bundle(page)[0])


def replace_anatomy_pair(source, pair):
    source = strip_report_synthesis(source)
    bundle, _ = source_bundle(source)
    bundle['anatomy'] = {**(bundle.get('anatomy') or {}), 'pair': pair}
    source = _replace_bundle(source, bundle)
    return refresh_anatomy_pair(source), with_neuron_labels(source, bundle)


def show_criticism_nav(source):
    """Keep the Criticism link visible in older reports at narrow widths."""
    return source.replace(",.topnav a[href='#criticism']{display:none}", "{display:none}")


def replace_ablation(source, section):
    source = strip_report_synthesis(source)
    bundle, _ = source_bundle(source)
    bundle['ablation'] = section
    # The original analysis predates the new measurements.
    (bundle.get('section_prose') or {}).pop('ablation', None)
    source = _replace_bundle(source, bundle)
    return show_criticism_nav(refresh_effect(source)), with_neuron_labels(source, bundle)


def apply_route_descriptions(bundle, result):
    """Use the prediction stage's names without changing semantic clusters."""
    names = result.get('route_names') or {}
    if isinstance(names, list):
        names = dict(zip(result.get('route_ids') or [], names))
    descriptions = {item['id']: item for item in result.get('descriptions') or []}
    for route in bundle.get('routes') or []:
        identifier = route.get('id')
        description = descriptions.get(identifier, {})
        name = description.get('name') or names.get(identifier)
        if name:
            route['name'] = name
        if 'description' in description:
            route['description'] = description['description']


def route_prediction_execution(result):
    execution = {'status': result.get('status', 'complete')}
    if 'reason' in result:
        execution['reason'] = result['reason']
    return execution


def update_route_prediction(source_html, result, *, interpretation_costs=None):
    """Add saved-attribution predictions while preserving the rest of the report."""
    matches = list(BUNDLE.finditer(source_html))
    if len(matches) != 1:
        raise ValueError('Expected exactly one report bundle')
    bundle = json.loads(matches[0][2])
    tests = copy.deepcopy(bundle.get('tests') or {})
    tests['route_prediction'] = copy.deepcopy(result)
    execution = dict(tests.get('execution') or {})
    execution['route_prediction'] = route_prediction_execution(result)
    tests['execution'] = execution
    bundle['tests'] = tests
    apply_route_descriptions(bundle, result)
    if interpretation_costs is not None:
        bundle['interpretation_costs'] = copy.deepcopy(interpretation_costs)
    return refresh_tests(_replace_bundle(source_html, bundle)), bundle


def source_bundle(source):
    matches = list(BUNDLE.finditer(source))
    if len(matches) != 1:
        raise ValueError('Expected exactly one report bundle')
    bundle = json.loads(matches[0][2])
    start = source.find('const B =')
    stop = source.find('const POS =',start)
    if start < 0 or stop < 0:
        raise ValueError('Unsupported report renderer bootstrap')
    bootstrap = source[start:stop]
    if 'const B = (() => {' in bootstrap:
        if "one_paragraph: '—'" not in bootstrap or 'saved.native' in bootstrap:
            raise ValueError('Unsupported activation-only report bootstrap')
        # Initialize the activation-only claim from the saved population statistics.
        bundle['neuron']['autointerp_label'] = None
        bundle['neuron']['autointerp_description'] = None
        tokens = bundle['distribution']['n_tokens']
        documents = (bundle.get('bigscan') or {}).get('n_docs')
        if documents is None:
            documents = (bundle.get('local_metadata') or {}).get('population_documents')
        bundle['claim'] = {'title':f"{bundle['neuron']['id']} · {tokens} saved tokens",
                          'lede':f'{tokens:,} tokens · {documents:,} documents · saved native gate/value/product.' if documents is not None else 'Saved native gate/value/product.',
                          'one_paragraph':'—','negative_pole':None,'what_would_change_my_mind':None,'confidence':None}
        bundle['criticism'] = []
        bundle['version'] = {**(bundle.get('version') or {}),'n':None,'ts':None,'history':[]}
        bundle['appendix'] = {**(bundle.get('appendix') or {}),'notes':[],'provenance':[]}
        if bundle.get('headline',{}).get('strip'):
            bundle['headline']['strip']['caption'] = 'Natural-corpus context at the strongest saved document peak.'
    elif not re.search(r'const B = JSON.parse\(document.getElementById\(["\x27]bundle["\x27]\).textContent\);',bootstrap):
        raise ValueError('Unsupported report bootstrap')
    synthesis = REPORT_SYNTHESIS.search(source)
    if synthesis is not None:
        data = json.loads(synthesis['data'])
        bundle['claim'] = data['claim']
        summaries = bundle.get('section_prose') or {}
        bundle['section_prose'] = {**summaries, **{
            key: {**(summaries[key] if isinstance(summaries.get(key), dict) else {}), **value}
            for key, value in data['section_prose'].items()
        }}
        bundle['synthesis_evidence'] = data['evidence']
    return bundle, (start,stop)

def strip_capture_table(source):
    source = re.sub(r'<table\b[^>]*>.*?</table>',
        lambda m:'' if re.search(r'capture\s+settings?',m[0],re.I) else m[0],source,flags=re.S|re.I)
    return re.sub(r'<h[1-6]\b[^>]*>\s*Capture settings?\s*</h[1-6]>', '', source,flags=re.I)


def presentation(source):
    """Shared report layout; short verdict badges, readable challenges and blank Appendix."""
    source = show_criticism_nav(source).replace("['Neuron paper',", "['Neuron report',")
    style = """<style id="neuron-paper-layout">
#criticism .crow > summary .pill{max-width:100%;white-space:nowrap}
@media(max-width:600px){
  #criticism .crow > summary{grid-template-columns:24px 90px minmax(0,1fr);gap:0 10px}
  #criticism .crow .bodyc{padding-left:0}
}
</style>"""
    source = re.sub(r'<style id="neuron-paper-layout">.*?</style>', '', source, flags=re.S)
    source = source.replace('</head>', style + '\n</head>', 1)
    appendix = """// 9. Appendix.
(() => { sec('appendix','Appendix',null,$('span',{class:'tile'},'—')); })();"""
    source = re.sub(r'// 9\. Appendix\.\s*\(\(\) => \{.*?\}\)\(\);',
                    lambda match: appendix, source, count=1, flags=re.S)
    return source

def readable_data_scripts(source):
    """Format embedded JSON data with indentation."""
    def pretty(match):
        data=json.loads(match.group(2))
        body=json.dumps(data,ensure_ascii=False,indent=2).replace('<',r'\u003c')
        return match.group(1)+'\n'+body+'\n'+match.group(3)
    return re.sub(r"""(<script\b[^>]*\btype\s*=\s*["']application/json["'][^>]*>)(.*?)(</script\s*>)""",pretty,source,flags=re.S|re.I)


def merge_test_fields(previous,current):
    merged={**copy.deepcopy(previous or {}),**copy.deepcopy(current or {})}
    merged['execution']={**(previous or {}).get('execution',{}),
                         **(current or {}).get('execution',{})}
    return merged

def merge(source, patches):
    bundle, _ = source_bundle(source)
    bundle.update(patches)
    if patches.get('ablation') is not None:
        source = refresh_effect(source)
    exemplars = bundle.get('exemplars') or {}
    if exemplars.get('bands'):
        bundle['exemplars'] = {**exemplars, 'bands': {
            band: sorted(samples, key=lambda sample: -abs(sample.get('activation') or 0))
            for band, samples in exemplars['bands'].items()
        }}
    search = bundle.get('search') or {}
    regions = search.get('regions', []) + search.get('regions_more', [])
    if regions:
        regions = sorted(regions, key=lambda region: -abs(region.get('best') or 0))
        visible = len(search.get('regions', []))
        bundle['search'] = {**search, 'regions': regions[:visible], 'regions_more': regions[visible:]}
    if bundle.get('clusters'):
        clusters = [{**cluster, 'examples': sorted(cluster.get('examples', []),
                     key=lambda example: -abs(example.get('activation') or 0))}
                    for cluster in bundle['clusters']]
        def cluster_strength(cluster):
            summary = cluster.get('response_summary') or {}
            values = [summary.get('min'), summary.get('max')]
            values.extend(example.get('activation') for example in cluster['examples'])
            return max(abs(value or 0) for value in values)
        bundle['clusters'] = sorted(clusters, key=cluster_strength, reverse=True)
    if (bundle.get('tests') or {}).get('route_prediction') is not None:
        tests = copy.deepcopy(bundle['tests'])
        result = tests['route_prediction']
        tests['execution'] = {**(tests.get('execution') or {}),
                              'route_prediction': route_prediction_execution(result)}
        bundle['tests'] = tests
        apply_route_descriptions(bundle, result)
        source = refresh_tests(source)
    bundle['neuron']={k:v for k,v in bundle.get('neuron',{}).items()
                      if k not in ['autointerp_label','autointerp_description']}
    source=source.replace("if (N.autointerp_label) meta.append($('span', null, 'Original autointerp label: ', $('b', null, N.autointerp_label))); ", '')
    source=source.replace('C.title || N.autointerp_label || N.id','C.title || N.id')
    # Unmeasured writer proposals stay in their saved experiment artifacts.
    if isinstance(bundle.get('tests'), dict) and 'plan' in bundle['tests']:
        tests = copy.deepcopy(bundle['tests'])
        tests.pop('plan')
        bundle['tests'] = tests or None
    from spectrum import anatomy_labels, render as render_spectrum
    anatomy_labels(bundle)
    source=render_spectrum(source)
    from saved_display import render as render_saved
    source=render_saved(source)
    # Layout transforms can change text before the bootstrap, including styles.
    # Locate it in the transformed source instead of reusing earlier offsets.
    start = source.find('const B =')
    stop = source.find('const POS =',start)
    source = source[:start]+"const B = JSON.parse(document.getElementById('bundle').textContent);\n\n"+source[stop:]
    source = _replace_bundle(source, bundle)
    if 'claim' in patches:
        source = re.sub(r'<title>.*?</title>','<title>'+html.escape(patches['claim']['title'])+'</title>',source,count=1,flags=re.S)
    # Remove embedded interpretation scripts after merging the final bundle.
    for sid in ['paper-review-data','paper-review-full-data','saved-interpretation','interpretation-adapter']:
        source = re.sub(r'<script\b(?=[^>]*\bid=["\x27]'+sid+r'["\x27])[^>]*>.*?</script>','',source,flags=re.S)
    graphs = bundle.get('graphs_curated') or {}
    picks = graphs if isinstance(graphs,list) else graphs.get('picks',[])
    relp_graphs = bool(picks) and all('RelP:' in x.get('graph',{}).get('meta',{}).get('method','') for x in picks)
    if relp_graphs:
        source = source.replace('Lines show total first-order source-to-root sensitivity; the slider sets how many are drawn.',
            'Edges are local RelP contributions weighted by receiver influence on the root; the slider sets how many are drawn.')
        source = source.replace('Edges are the strongest attribution links among the visible nodes; the slider sets how many are drawn.',
            'Edges are local RelP contributions weighted by receiver influence on the root; the slider sets how many are drawn.')
        source = source.replace('of ${el.length} source-to-root lines','of ${el.length} edges')
        source = source.replace("class: 'filt' }, 'Lines '","class: 'filt' }, 'Edges '")
        source = source.replace('none among visible source-to-root lines','none among visible edges')
        source = source.replace('Node size follows total first-order influence on the root; lines do not isolate direct neuron-to-neuron edges. The thick black ring marks this neuron.',
            'Node size follows RelP influence on the root. Edges show signed flow between neurons; the thick black ring marks this neuron.')
    source = source.replace("'Attribution graphs'", "'Three graphs'")
    source = source.replace("'circuit','tests'];", "'circuit','tests','appendix','criticism'];")
    source=re.sub(r"(if\(node.textContent!=='One sentence, three signals'\)return;)(?:\s*if\(B\.anatomy\?\.strip\?\.case_id\)return;)*",
        r"\1 if(B.anatomy?.strip?.case_id)return;", source)
    source=source.replace("|| {}).name || String(id);","|| {}).name || (B.anatomy?.group_names || {})[id] || String(id);")
    adapter='''<script id="paper-section-prose">
(() => {
  const names={spectrum:'spectrum',search:'search',clusters:'clusters',anatomy:'anatomy',ablation:'ablation',family:'family',circuit:'circuit',tests:'tests'};
  const prose={...(B.section_prose || {})};
  // Earlier reports placed output-effect prose next to a copied logit table in Circuit.
  const legacyOutput=B.circuit?.logits && !B.circuit?.upstream?.length && !B.circuit?.downstream?.length;
  if(legacyOutput && prose.circuit && !prose.ablation){prose.ablation=prose.circuit;delete prose.circuit;}
  Object.entries(prose).forEach(([key,p])=>{
    if(key==='tests' && !['blind_max','blind_min','route_prediction','steering','ablation'].some(k=>has(B.tests?.[k])))return;
    const section=document.getElementById(names[key]); if(!section)return;
    const summary=typeof p==='string'?p:p.summary;
    if(summary){
      let why=section.querySelector('.why');
      if(!why){why=$('div',{class:'why'});section.querySelector('.text').append(why);}
      why.replaceChildren(P(summary));
    }
    if(key==='anatomy' && p.strip_caption) section.querySelectorAll('figure').forEach(f=>{
      if(f.querySelector('.sub')?.textContent==='One sentence, three signals'){
        const cap=f.querySelector('.cap');if(cap)cap.textContent=p.strip_caption;
      }
    });
  });
  if(B.family?.measurements?.length && !B.section_prose?.family?.summary){
    const why=document.querySelector('#family .why');
    if(why)why.replaceChildren(P(B.family.source_kind==='saved_coactivation'?B.family.method:'Joint native activations of the target and selected neighbours on the same inputs. Links show the measured local MLP contributions.'));
  }
  if(B.circuit?.method){
    const why=document.querySelector('#circuit .why');
    if(why && !B.section_prose?.circuit?.summary)why.replaceChildren(P(
      B.circuit.method.startsWith('RelP:') ?
      'Local RelP contributions, pooled over the selected atlas. Downstream coverage is the nominated readers at the anchor position.' :
      B.circuit.method));
  }
  if(B.tests?.explanation_version){
    const why=document.querySelector('#tests .why');
    if(why && !B.section_prose?.tests?.summary)why.replaceChildren(P(B.tests.reused_measurement?'Tests reused from an earlier report; this run does not retest its new explanation.':'Tests of the frozen initial explanation, followed by report revision.'));
  }
})();
</script>'''
    source=re.sub(r'<script id="paper-section-prose">.*?</script>','',source,flags=re.S)
    source=source.replace('</body>',adapter+'\n</body>') if '</body>' in source else source+adapter
    source=re.sub(r'<!-- paper-review:start -->.*?<!-- paper-review:end -->','',source,flags=re.S)
    from report_layout import render as align_layout
    from exemplar_display import render as render_exemplars
    source = presentation(render_exemplars(align_layout(source)))
    if (patches.get('anatomy') or {}).get('pair') is not None:
        source = refresh_anatomy_pair(source)
    return readable_data_scripts(strip_capture_table(source)), bundle
