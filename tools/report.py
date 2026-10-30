"""Merge measured artifacts into a new original-style report copy."""
import copy
import html
import json
import re
import shutil
from pathlib import Path
from common import digest, model_config, read, write, now

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
    source = source.replace("const eyebrow = ['Neuron paper',", "const eyebrow = ['Neuron report',")
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

def run(request,output):
    output, report = Path(output), Path(request['report']).resolve()
    source = (report/'index.html').read_text()
    data = read(report/'report.json')
    from spectrum_sampling import register_contexts
    register_contexts(data, request.get('sections', []))
    identity = data['identity']
    neuron = {'layer':identity['layer'],'index':identity['neuron']}
    path,config = model_config(request['model'],neuron)
    patches, sources = {}, []
    native_tests, steering_tests = [], []
    refreshed_test_fields = set()
    effective,_ = source_bundle(source)
    from spectrum import scan_fields
    patches.update(scan_fields(effective,data))
    if request.get('stage'):
        prior_version=effective.get('version') or {}
        number=2 if request['stage']=='03-revision' else 1
        patches['version']={'n':number,'ts':now(),'history':[]}
    annotations=read(request['annotations']) if request.get('annotations') else {}
    selection={**effective.get('display_selection',{})}
    for key in ['headline_figure_id','graph_ids']:
        if key in annotations:selection[key]=annotations[key]
    if selection:patches['display_selection']=selection
    applied_annotations={'figures':set(),'graphs':set()}
    if request.get('interpretation'):
        interpretation = Path(request['interpretation']).resolve()
        obj = read(interpretation/'output.json')
        patches.update({k:obj[k] for k in ['claim','criticism','agent_read'] if k in obj})
        sources.append(str(interpretation))
    artifacts = output/'artifacts'
    artifacts.mkdir()
    for key, field, filename in [('sweeps','custom','figure.json'),('graphs','graphs_curated','graph.json')]:
        measured = []
        seen = set()
        for folder in request.get(key,[]):
            folder = Path(folder).resolve()
            result = read(folder/'result.json')
            if not (folder/filename).exists():
                continue
            payload = read(folder/filename)
            if key=='graphs':
                ab=result.get('ablation') or {}
                if ab.get('status')=='complete':
                    native_tests.append({'graph_id':payload['id'],'effect':ab['delta_target_logit'],
                        'note':'Zero product at token '+str(ab['position'])+'; change in logit of '+
                        repr(ab['reference_token'])+'; KL='+format(ab['kl'],'.6g')+'.'})
                if result.get('steering'):steering_tests.append(result['steering'])
            if payload['id'] in seen:
                continue
            seen.add(payload['id'])
            section='figures' if key=='sweeps' else 'graphs'
            if payload['id'] in annotations.get(section,{}):
                prose=['title','rationale','caption'] if section=='figures' else ['title','why','look_at']
                payload.update({k:v for k,v in annotations[section][payload['id']].items() if k in prose})
                applied_annotations[section].add(payload['id'])
            measured.append(payload)
            target=artifacts/(key+'-'+payload['id'])
            shutil.copytree(folder,target)
            sources.append(str(folder))
        if measured:
            items_key='figures' if key=='sweeps' else 'picks'
            previous=effective.get(field) or {}
            if isinstance(previous,list): previous={items_key:previous}
            combined={x['id']:x for x in previous.get(items_key,[])}
            combined.update({x['id']:x for x in measured})
            patches[field]={**previous,items_key:list(combined.values())}
    from paper_sections import labelled
    for number,folder in enumerate(request.get('sections',[])):
        folder=Path(folder)
        section_data=read(folder/'sections.json')
        source_request=folder/'tool-request.json'
        if source_request.exists() and read(source_request).get('only')=='route_prediction':
            refreshed_test_fields.add('route_prediction')
        for key,value in section_data.items():
            # An omitted measurement round must not erase an earlier measured family.
            if key=='family' and not value.get('measurements'):
                continue
            if key=='ablation':
                from followups import combine_ablations
                value=combine_ablations(patches.get(key) or effective.get(key),value)
            if key=='tests':
                value=merge_test_fields(patches.get(key) or effective.get(key),value)
            if key=='search' and key in patches:
                old=patches[key]
                regions=old.get('regions',[])+value.get('regions',[])
                value={**value,'regions':regions,'n_regions':len(regions),
                       'n_texts':old.get('n_texts',0)+value.get('n_texts',0),
                       'best_positive':max([x for x in [old.get('best_positive'),value.get('best_positive')] if x is not None],default=None),
                       'best_negative':min([x for x in [old.get('best_negative'),value.get('best_negative')] if x is not None],default=None)}
            patches[key]=value
        target=artifacts/('sections-'+str(number+1))
        shutil.copytree(folder,target)
        sources.append(str(folder))
    if native_tests or steering_tests:
        tests=copy.deepcopy(patches.get('tests') or effective.get('tests') or {})
        if native_tests:tests['ablation']=native_tests
        if steering_tests:tests['steering']=steering_tests
        patches['tests']=tests
    if annotations.get('routes') or annotations.get('units'):
        combined={**effective,**patches}
        named=labelled(combined,annotations)
        for key in ['routes','clusters','family','circuit']:
            if key in named:patches[key]=named[key]
    from semantic_clusters import apply as apply_semantic_clusters
    cluster_definitions = annotations.get('clusters', effective.get('cluster_definitions'))
    if cluster_definitions is not None:
        combined = {**effective, **patches}
        apply_semantic_clusters(combined, cluster_definitions, request.get('sections', []))
        for key in ['clusters', 'cluster_definitions', 'cluster_source']:
            if key in combined: patches[key] = combined[key]
    if annotations.get('anatomy'):
        from paper_sections import prepare
        patches['anatomy']=prepare({'anatomy':annotations['anatomy']},report)['_anatomy']
    if annotations.get('sections'):
        patches['section_prose']={**effective.get('section_prose',{}),**annotations['sections']}

    # A saved section summary predates a partial test rerun. Render the measured
    # table with the standard introduction instead of retaining stale conclusions.
    if refreshed_test_fields:
        prose={**effective.get('section_prose',{}),**patches.get('section_prose',{})}
        prose.pop('tests',None)
        patches['section_prose']=prose

    from saved_display import peak_fields, default_anatomy, sweep_search
    if request.get('interpretation') and (interpretation/'input.json').is_file():
        patches.update(peak_fields(read(interpretation/'input.json')))
    combined = {**effective, **patches}
    if not annotations.get('anatomy'):
        anatomy = default_anatomy(combined, data)
        if anatomy is not None:
            patches['anatomy'] = anatomy
    custom = combined.get('custom') or {}
    figures = custom if isinstance(custom, list) else custom.get('figures', [])
    search = sweep_search(request.get('sweeps', []), combined.get('search'), figures)
    if search is not None:
        patches['search'] = search
    combined = {**effective, **patches}
    from report_layout import headline
    selected_headline = headline(combined, report, request.get('sweeps', []), data)
    if selected_headline is not None:
        patches['headline'] = selected_headline

    if request.get('annotations'):shutil.copyfile(request['annotations'],output/'annotations.json')
    from report_completion import complete
    combined=copy.deepcopy({**effective,**patches})
    complete(combined,request,data)
    for field in ['spectrum','anatomy','family','circuit','routes','clusters','agent_read_full','graphs_curated','tests']:
        if field in combined:patches[field]=combined[field]
    page,bundle = merge(source,patches)
    (output/'index.html').write_text(page)
    write(output/'paper-data.json',{'bundle':bundle})
    # Keep the original local ancillary data for relative links, without copying previous interpretation outputs.
    write(output/'report.json',data)
    for name in ['dataset-index.json','token-rankings.json','effects.json',
                 'peak-census.json','peak-census-negative.json','peak-census-absolute.json',
                 'anatomy-scan-evidence.json']:
        if (report/name).is_file():
            shutil.copyfile(report/name,output/name)
    write(output/'publication.json',{'source_report':str(report),'source_html_sha256':digest(report/'index.html'),
          'changed_bundle_fields':sorted(patches),'experiment_sources':sources,'capture_setting_table':False})
    result = {'tool':'publish_report','status':'complete','html':str(output/'index.html'),
              'changed_bundle_fields':sorted(patches),'new_model_forwards':0,'llm_requests':0}
    write(output/'result.json',result)
    return result


def refresh_review(report,review,output):
    """Create a report copy changing only its independent reading and label renderer."""
    import os
    from paper_reading import display_names
    from report_completion import full_reading
    report,output=Path(report),Path(output)
    shutil.copytree(report,output,copy_function=os.link)
    reading=display_names(review)
    patch={'agent_read':reading}
    full_reading(patch)
    page,bundle=merge((report/'index.html').read_text(),patch)
    temp=output/'index.html.tmp'
    temp.write_text(page)
    temp.replace(output/'index.html')
    write(output/'paper-data.json',{'bundle':bundle})
    write(output/'publication.json',{
        'source_report':str(report),'source_html_sha256':digest(report/'index.html'),
        'changed_bundle_fields':['agent_read','agent_read_full'],
        'capture_setting_table':False})
    result={'tool':'refresh_independent_review','status':'complete',
            'html':str(output/'index.html'),'new_model_forwards':0,'llm_requests':0}
    write(output/'result.json',result)
    return result


def refresh_layout(report,output,sections=(),bundle_patch=None):
    """Render saved data, restoring measured sections from their existing artifacts."""
    import os
    report,output=Path(report),Path(output)
    shutil.copytree(report,output,copy_function=os.link)
    saved,_=source_bundle((report/'index.html').read_text())
    patch=copy.deepcopy(bundle_patch or {})
    if saved.get('headline') and (report/'report.json').is_file():
        from report_layout import headline
        patch['headline']=headline({**saved,**patch},report,sections)
    for folder in sections:
        section_data=read(Path(folder)/'sections.json')
        tests=section_data.get('tests')
        if tests:patch['tests']=merge_test_fields(patch.get('tests') or saved.get('tests'),tests)
        family=section_data.get('family')
        if family and family.get('measurements'):patch['family']=copy.deepcopy(family)
    if 'family' in patch and (report/'annotations.json').is_file():
        from paper_sections import labelled
        patch['family']=labelled({'family':patch['family']},read(report/'annotations.json'))['family']
    definitions=patch.get('cluster_definitions',saved.get('cluster_definitions'))
    if definitions is not None and sections:
        from semantic_clusters import apply as apply_semantic_clusters
        combined={**saved,**patch}
        apply_semantic_clusters(combined,definitions,sections)
        for key in ['clusters','cluster_definitions','cluster_source']:
            if key in combined:patch[key]=combined[key]
    from report_completion import measured_contexts
    combined=copy.deepcopy({**saved,**patch})
    measured_contexts(combined,report)
    for key in ['graphs_curated','tests']:
        if combined.get(key)!=saved.get(key):patch[key]=combined[key]
    page,bundle=merge((report/'index.html').read_text(),patch)
    temporary=output/'index.html.tmp'
    temporary.write_text(page)
    temporary.replace(output/'index.html')
    write(output/'paper-data.json',{'bundle':bundle})
    write(output/'publication.json',{'source_report':str(report),
        'source_html_sha256':digest(report/'index.html'),'changed_bundle_fields':sorted(patch),
        'saved_test_sections':[str(p) for p in sections],
        'capture_setting_table':False})
    result={'tool':'refresh_report_layout','status':'complete',
        'html':str(output/'index.html'),'new_model_forwards':0,'llm_requests':0}
    write(output/'result.json',result)
    return result
