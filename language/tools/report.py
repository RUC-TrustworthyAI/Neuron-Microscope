"""Merge measured artifacts into a new original-style report copy."""
import copy
import html
import json
import re
from pathlib import Path

BUNDLE = re.compile(r'(<script\b(?=[^>]*\bid=["\x27]bundle["\x27])[^>]*>)(.*?)(</script>)', re.S)

def source_bundle(source):
    match = BUNDLE.search(source)
    if match is None or len(list(BUNDLE.finditer(source))) != 1:
        raise ValueError('Expected exactly one report bundle')
    bundle = json.loads(match[2])
    start = source.find('const B =')
    stop = source.find('const POS =',start)
    if start < 0 or stop < 0:
        raise ValueError('Unsupported report renderer bootstrap')
    bootstrap = source[start:stop]
    if 'const B = (() => {' in bootstrap:
        if "one_paragraph: '—'" not in bootstrap or 'saved.native' in bootstrap:
            raise ValueError('Unknown report bootstrap; refusing to resurrect hidden interpretations')
        # Match the effective activation-only view rather than its hidden historical bundle prose.
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
    return bundle, (start,stop)

def strip_capture_table(source):
    source = re.sub(r'<table\b[^>]*>.*?</table>',
        lambda m:'' if re.search(r'capture\s+settings?',m[0],re.I) else m[0],source,flags=re.S|re.I)
    return re.sub(r'<h[1-6]\b[^>]*>\s*Capture settings?\s*</h[1-6]>', '', source,flags=re.I)


def presentation(source):
    """Shared report layout; short verdict badges, readable challenges and blank Appendix."""
    source = source.replace("['Neuron paper',", "['Neuron report',")
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
    """Keep self-contained HTML data readable without single-line multi-MB tool output."""
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
    bundle, (start,stop) = source_bundle(source)
    bundle.update(patches)
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
    # Route prediction is disabled; retain raw artifacts and leave the report slot empty.
    tests=copy.deepcopy(bundle.get('tests') or {})
    had_route_score=tests.get('route_prediction') is not None
    tests['route_prediction']=None
    tests.setdefault('execution',{})['route_prediction']={'status':'not_run','reason':'Route prediction is disabled in this pipeline.'}
    bundle['tests']=tests
    if had_route_score:
        # The earlier combined summary includes the now-hidden route score.
        prose=copy.deepcopy(bundle.get('section_prose') or {})
        prose.pop('tests',None)
        bundle['section_prose']=prose
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
    match = BUNDLE.search(source)
    encoded = json.dumps(bundle,ensure_ascii=False,allow_nan=False,separators=(',',':')).replace('<','\\u003c').replace('\u2028','\\u2028').replace('\u2029','\\u2029')
    source = source[:match.start(2)]+encoded+source[match.end(2):]
    if 'claim' in patches:
        source = re.sub(r'<title>.*?</title>','<title>'+html.escape(patches['claim']['title'])+'</title>',source,count=1,flags=re.S)
    # Historical interpretation adapters must not override the now-explicit final bundle.
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
    return readable_data_scripts(strip_capture_table(presentation(render_exemplars(align_layout(source))))), bundle
