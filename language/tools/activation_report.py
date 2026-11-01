"""Initialize an activation-only report from saved measurements."""
def initial_html(source,data=None):
    from report import source_bundle,merge
    bundle,_=source_bundle(source)
    neuron={**bundle['neuron'],'autointerp_label':None,'autointerp_description':None}
    tokens=bundle['distribution']['n_tokens']
    documents=(bundle.get('bigscan') or {}).get('n_docs')
    if documents is None:documents=(bundle.get('local_metadata') or {}).get('population_documents')
    claim={'title':f"{neuron['id']} · {tokens} saved tokens",
           'lede':f'{tokens:,} tokens · {documents:,} documents · saved native gate/value/product.' if documents is not None else 'Saved native gate/value/product.',
           'one_paragraph':'—','negative_pole':None,'what_would_change_my_mind':None,'confidence':None}
    from spectrum import scan_fields
    scan=scan_fields(bundle,data) if data else {}
    anatomy=__import__('copy').deepcopy(bundle.get('anatomy') or {})
    if anatomy.get('strip'):
        for key in ['case_id','source_start','anchor_index']:anatomy['strip'].pop(key,None)
        anatomy['strip']['caption']='Saved native gate, value and product.'
    anatomy['points']=[p for p in anatomy.get('points',[]) if p.get('source_family')!='probe' and p.get('source')!='corpus:scan']
    anatomy['n']=len(anatomy['points'])
    anatomy.pop('selection',None)
    anatomy.pop('groups',None)
    for key in ['clusters','semantic_groups','semantic_labeler']:anatomy.pop(key,None)
    anatomy.pop('quadrants',None)
    anatomy['headline']=[]
    anatomy.pop('pole_strips',None)
    anatomy.pop('headline_source',None)
    anatomy['group_names']={}
    for point in anatomy.get('points',[]):
        point['cluster']='unassigned'
        for key in ['semantic_group','label_source','label_reason','predicted_pole','prediction_reason','explanation_version','evaluation_split']:point.pop(key,None)
    for key in ['prediction_summary','explanation_version','colour_mode']:anatomy.pop(key,None)
    page,_=merge(source,{**scan,'neuron':neuron,'claim':claim,'criticism':[],
        'agent_read':None,'agent_read_full':None,'interpretation_costs':None,'spectrum':None,'clusters':[],'routes':[],'routes_matrix':None,'routes_meta':None,
        'cluster_source':None,'cluster_definitions':[],'section_prose':{},'family':None,'circuit':None,'tests':None,
        'custom':None,'graphs_curated':None,'anatomy':anatomy,
        'headline':{'strip':bundle.get('headline',{}).get('strip')},
        'version':{'n':None,'ts':None,'history':[]},'appendix':{'notes':[],'provenance':[]}})
    return page
