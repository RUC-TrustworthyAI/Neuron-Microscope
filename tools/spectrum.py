"""Label saved contexts and build the activation spectrum without model inference."""
import bisect
import collections
import concurrent.futures
import copy
import math
from pathlib import Path
from common import read, write, now

def window(data, case_id, before=512, after=10):
    point = data['points'][str(case_id)]
    doc = data['documents'][point['document']]
    pos = point['position']
    lo, hi = max(0, pos-before), min(len(doc['tokens']), pos+after+1)
    trace = [{'token':doc['tokens'][i], 'activation':doc['product'][i]} for i in range(lo, hi)]
    return {'id':str(case_id), 'case_id':str(case_id), 'text':''.join(doc['tokens'][lo:hi]),
            'activation':point['values'][2], 'token_activations':trace,
            'peak_index':pos-lo, 'cut_left':lo>0, 'cut_right':hi<len(doc['tokens']),
            'source':'saved natural-corpus token', 'document':point['document']}

def scan_fields(bundle, data):
    scan = copy.deepcopy(bundle.get('bigscan'))
    if scan is None or not data.get('examples'):
        return {}
    scan['top'] = [window(data, p) for p in data['examples']['top_documents'][:6]]
    scan['bottom'] = [window(data, p) for p in data['examples']['bottom_documents'][:4]]
    return {'bigscan':scan}

def anatomy_labels(bundle):
    spectrum = bundle.get('spectrum') or {}
    anatomy = bundle.get('anatomy')
    if not anatomy or not spectrum.get('points'):
        return
    if anatomy.get('semantic_labeler') and anatomy.get('clusters'):
        return
    labels = {str(p['id']):p['group'] for p in spectrum['points']}
    quadrants = collections.defaultdict(collections.Counter)
    for point in anatomy.get('points', []):
        group = (point['cluster'] if point.get('source_family')=='probe'
                 and (point.get('cluster')=='unlabelled-probes' or point.get('label_source')=='existing')
                 else labels.get(str(point['id']), 'unlabelled'))
        point['cluster'] = group
        quadrant = ('g+' if point['gate'] > 0 else 'g-') + ('u+' if point['up'] > 0 else 'u-')
        quadrants[group][quadrant] += 1
    anatomy['quadrants'] = {k:dict(v) for k,v in quadrants.items()}
    anatomy['group_names'] = {g['id']:g['name'] for g in spectrum.get('groups',[])}
    if any(p.get('cluster')=='unlabelled-probes' for p in anatomy.get('points',[])):
        anatomy['group_names']['unlabelled-probes']='Unlabelled experimental contexts'

def cases(data, tokenizer):
    memberships = collections.defaultdict(list)
    for name, ids in data['examples'].items():
        for p in ids: memberships[str(p)].append(name)
    for band in data['bands']:
        for p in band['points']: memberships[str(p)].append('p%s-p%s' % (band['low'],band['high']))
    result = []
    for key, point in sorted(data['points'].items(), key=lambda item:int(item[0])):
        doc = data['documents'][point['document']]
        pos = point['position']
        result.append({'id':str(key), 'sample_groups':memberships.get(str(key),['scatter']),
            'anchor':{'token':doc['tokens'][pos], 'position':pos,
                      'gate':point['values'][0], 'value':point['values'][1], 'product':point['values'][2]},
            'observed_prefix':tokenizer.decode(doc['token_ids'][:pos+1],skip_special_tokens=False),
            'following_prediction_target':tokenizer.decode(doc['token_ids'][pos+1:pos+11],skip_special_tokens=False)})
    return result

def build(data, groups, assignments):
    values = [p['values'][2] for p in data['points'].values()]
    bound = 2.0 ** math.ceil(math.log2(max(map(abs, values),default=1) or 1))
    small = bound / 256
    powers = [small * 2**i for i in range(9)]
    negative = [-p for p in reversed(powers) if p/2 < abs(min(min(values),0)) or p == small]
    positive = [p for p in powers if p/2 < max(max(values),0) or p == small]
    edges = sorted(set(negative+positive))
    bins = [{'i':i,'label':f'{a:g}…{b:g}','n_labeled':0,'groups':{}} for i,(a,b) in enumerate(zip(edges,edges[1:]))]
    points = []
    counts = collections.Counter()
    for key, point in data['points'].items():
        group = assignments.get(str(key),{}).get('group','unlabelled')
        i = max(0,min(len(bins)-1,bisect.bisect_right(edges,point['values'][2])-1))
        row = window(data,key)
        row.update(bin=i,group=group)
        points.append(row)
        bins[i]['n_labeled'] += 1
        bins[i]['groups'][group] = bins[i]['groups'].get(group,0)+1
        counts[group] += 1
    definitions = [dict(g,n=counts[g['id']]) for g in groups]
    if counts['unlabelled']:
        definitions.append({'id':'unlabelled','name':'Unlabelled','description':'No resolved semantic assignment.','pole':'either','n':counts['unlabelled']})
    return {'edges':edges,'groups':definitions,'bins':bins,'points':points,
            'labeler':'DeepSeek high','sample_hist':{'counts':[b['n_labeled'] for b in bins]},
            'sampling':'All unique saved report positions: extremes, quantile examples and scatter samples. These are sample counts, not estimated semantic frequencies across the full corpus.',
            'n_requested':len(points),'n_assigned':len(points)-counts['unlabelled']}

def run(session, arguments, llm=None):
    from tokenizers import Tokenizer
    from blind_tests import call
    if llm is None: llm=call
    number=1+len(list(session.root.glob('spectrum-*')))
    folder=session.root/('spectrum-%03d'%number);folder.mkdir()
    data=read(Path(session.config['report'])/'report.json')
    prior={}
    groups=arguments.get('groups')
    resume=arguments.get('resume_from')
    if resume:
        previous=session.root/resume
        groups=read(previous/'groups.json')
        prior=read(previous/'assignments.json')
        sources=[str(previous)]
        for candidate in sorted(session.root.glob('spectrum-*')):
            if candidate==folder or not (candidate/'assignments.json').exists():continue
            if read(candidate/'groups.json')==groups:
                for key,value in read(candidate/'assignments.json').items():prior.setdefault(key,value)
                sources.append(str(candidate))
        write(folder/'resume-source.json',{'directories':sources,'n_retained':len(prior)})
    allowed={g['id'] for g in groups}
    tokenizer=Tokenizer.from_file(str(Path(session.config['model'])/'tokenizer.json'))
    inputs=cases(data,tokenizer)
    write(folder/'cases.json',inputs);write(folder/'groups.json',groups)
    groups_json=__import__('json').dumps(groups,ensure_ascii=False)
    def label_batch(item):
        index, batch=item
        out=folder/('batch-%03d'%index)
        messages=[
          {'role':'system','content':'Classify each saved neuron activation context into one of the supplied semantic groups. Read the observed prefix through the anchor token. Following text is a prediction target, not prior context. Use the group definitions and the context together; a sign or magnitude alone does not identify its meaning. Return JSON: {"labels":[{"id":"case ID","group":"group ID","reason":"short explanation"}]}.'},
          {'role':'user','content':'Semantic groups:\n'+groups_json+'\n\nSaved cases:\n'+__import__('json').dumps(batch,ensure_ascii=False)}]
        try:
            answer=llm(messages,out,'spectrum-labeler',max_tokens=393216)
            labels=answer.get('labels',[])
            if isinstance(labels,dict):labels=[{'id':k,**(v if isinstance(v,dict) else {'group':v})} for k,v in labels.items()]
            ids={c['id'] for c in batch}
            return {str(x['id']):x for x in labels if str(x.get('id')) in ids and x.get('group') in allowed}
        except Exception as exc:
            out.mkdir(parents=True,exist_ok=True)
            write(out/'failure.json',{'error_type':type(exc).__name__,'error':str(exc)})
            return {}
    remaining=[case for case in inputs if case['id'] not in prior]
    batches=list(enumerate([remaining[i:i+24] for i in range(0,len(remaining),24)],1))
    assignments=dict(prior)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(label_batch,batches):assignments.update(result)
    write(folder/'assignments.json',assignments)
    spectrum=build(data,groups,assignments)
    from group_poles import POLES, annotate
    if any(g.get('pole') not in POLES for g in spectrum['groups']):
        spectrum=annotate(spectrum,folder/'group-poles',llm=llm)
        write(folder/'groups.json',spectrum['groups'])
    from saved_analysis import semantic_clusters
    write(folder/'sections.json',{'spectrum':spectrum,'clusters':semantic_clusters(spectrum),
                                 'cluster_source':'saved_semantic_groups'})
    identity=folder.name
    req={'id':identity,'title':'Semantic activation spectrum','output':str(folder),'report':session.config['report']}
    write(folder/'tool-request.json',req)
    result={'status':'complete' if spectrum['n_assigned']==len(inputs) else 'partial',
            'sections':['spectrum'],'n_requested':len(inputs),'n_assigned':spectrum['n_assigned'],
            'groups':[{'id':g['id'],'name':g['name'],'n':g['n']} for g in spectrum['groups']],
            'resume_from':identity,'missing_case_ids':[p['id'] for p in spectrum['points'] if p['group']=='unlabelled'],
            'batch_count':len(batches),'n_retained':len(prior),'n_new_requested':len(remaining),'new_model_forwards':0,'finished_at':now()}
    write(folder/'result.json',result)
    session.jobs.append({'tool':'paper_sections','request':req})
    write(session.root/'experiment-index.json',session.jobs)
    return result

def _legacy_render(source):
    if '<!-- native-spectrum-render-v1 -->' in source:
        return source
    source=source.replace("if (BS.max != null) t2.append(kTile('scan max vs harvest'",
                          "if (BS.max != null && BS.harvest_max != null) t2.append(kTile('scan max vs harvest'")
    source=source.replace("if (BS.min != null) t2.append(kTile('scan min vs harvest'",
                          "if (BS.min != null && BS.harvest_min != null) t2.append(kTile('scan min vs harvest'")
    source=source.replace("const toksA = (w.tokens || []).map",
                          "const toksA = w.token_activations || (w.tokens || []).map")
    source=source.replace('Windows from the scan; only the peak token has a value, so it alone is shaded.',
                          'Saved scan windows with measured activations at every displayed token.')
    source=source.replace("hist = (S.corpus_hist || {}).counts;",
                          "hist = (S.sample_hist || S.corpus_hist || {}).counts;")
    source=source.replace('const colour = polePalette(G); SPEC_COLOUR = colour;',
                          'const sampled = !!S.sample_hist; const colour = polePalette(G); SPEC_COLOUR = colour;')
    source=source.replace("mode === 'counts' ? 'tokens' : 'share'",
                          "mode === 'counts' ? (sampled ? 'samples' : 'tokens') : 'share'")
    source=source.replace('Cn != null ? `<br>≈ ${Math.round(sh * Cn).toLocaleString()} corpus tokens` : \'\'',
                          'Cn != null ? (sampled ? `<br>${n} saved samples` : `<br>≈ ${Math.round(sh * Cn).toLocaleString()} corpus tokens`) : \'\'')
    source=source.replace("mk('counts', 'Counts, log')","mk('counts', sampled ? 'Sample counts, log' : 'Counts, log')")
    old="const cap = `Groups defined by Fable; labels assigned by ${S.labeler || 'Sonnet'} labelers.` + ((S.corpus_hist || {}).source ? ` Corpus counts from a ${S.corpus_hist.source}.` : '') + ' In the counts view, bar height is the corpus count on a log axis and the segments are the labeled shares, so segment proportions match the share view and only bar heights differ.';"
    new="const cap = `Semantic groups assigned by ${S.labeler || 'the labeler'}.` + (sampled ? ' ' + S.sampling : ' Corpus counts and labelled shares are displayed separately.');"
    source=source.replace(old,new)
    source=source.replace('let SPEC_COLOUR = {};', '')
    source=source.replace('const clusterColour = id => {',
                          'let SPEC_COLOUR = {};\nconst clusterColour = id => { if (SPEC_COLOUR[id]) return SPEC_COLOUR[id];')
    return source + '\n<!-- native-spectrum-render-v1 -->'


def render(source):
    source=_legacy_render(source)
    source=source.replace('hist = (S.sample_hist || S.corpus_hist || {}).counts;',
                          'hist = (S.corpus_hist || S.sample_hist || {}).counts;')
    source=source.replace('const sampled = !!S.sample_hist;',
                          'const sampled = !S.corpus_hist && !!S.sample_hist;')
    source=source.replace("sampled ? 'Sample counts, log' : 'Counts, log'",
                          "sampled ? 'Sample counts, log' : 'Corpus counts, log'")
    source=source.replace('Cn != null ? (sampled ? `<br>${n} saved samples` : `<br>≈ ${Math.round(sh * Cn).toLocaleString()} corpus tokens`) : \'\'',
                          'Cn != null ? `<br>${n} labelled samples in this group/bin<br>${Cn.toLocaleString()} ${sampled ? "saved samples" : "corpus tokens"} in this bin` : \'\'')
    source=source.replace("' Corpus counts and labelled shares are displayed separately.'",
                          "' Bar heights count all saved corpus tokens; colours show group shares among selected labelled examples in each bin.'")
    if "class: 'unlabelled-corpus'" not in source:
        empty_bar="""const Cn = hist ? hist[i] : null;
      if (nl === 0 && mode === 'counts' && Cn > 0) {
        const r=svgEl('rect',{class:'unlabelled-corpus',x:x0,y:yC(Cn),width:bw,height:base-yC(Cn),fill:'url(#hatch)','data-bin':i,'data-corpus-count':Cn});
        r.onmousemove=ev=>showTip(ev,`<b>${Cn.toLocaleString()} corpus tokens</b><br>No saved semantic labels in this bin.`);
        r.onmouseleave=hideTip;r.onclick=()=>{selBin=selBin===i?null:i;shown=12;update();};layer.append(r);
      }
      if (nl > 0"""
        source=source.replace('const Cn = hist ? hist[i] : null; if (nl > 0',empty_bar)
    return source
