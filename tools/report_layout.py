"""Keep generated reports in the reference paper's section structure."""
import copy
import json
from paper_reading import GROUP_LABELS
import re
from pathlib import Path
from common import read

COLOURS = "// Group colours use explicit semantic response poles, as in the reference renderer.\nconst poleOf = g => typeof g.pole === 'number' ? (g.pole > 0 ? 'positive' : g.pole < 0 ? 'negative' : 'either') : (g.pole || 'either');\nconst isOther = g => /^other$/i.test(String(g.id || '')) || /^other$/i.test(g.name || '');\n"

CIRCUIT = """// 6. Circuit tables: upstream writers and downstream readers.
(() => {
  const G = B.circuit || {};
  const table = (items, caption) => {
    if (!has(items)) return $('span', {class:'tile'}, '—');
    const scale = Math.max(...items.map(u => Math.abs(u.weight || 0)), 1e-9);
    const counted = items.some(u => u.count != null);
    const rows = $('table', null, $('tr', null, $('th',null,'Unit'), $('th',null,'Label'),
      $('th',{style:'text-align:right'},'Weight'), counted ? $('th',{style:'text-align:right'},'Graphs') : null));
    const row = u => $('tr',null, $('td',{class:'id'},unitId(u)), $('td',null,u.label || ''),
      $('td',{class:'num'},tile(u.weight,{signed:true,scale,d:2})),
      counted ? $('td',{class:'num'},u.count ?? '—') : null);
    items.slice(0,12).forEach(u => rows.append(row(u)));
    let more = null;
    if (items.length > 12) {
      more = $('button',{class:'more'},'Show '+(items.length-12)+' more');
      more.onclick = () => { items.slice(12).forEach(u => rows.append(row(u))); more.remove(); };
    }
    return fig($('div',null,rows,more),caption);
  };
  const columns = $('div',{class:'cols'},
    $('div',null,$('div',{class:'sub'},'Upstream'),table(G.upstream,'Upstream units, aggregated over the atlas graphs.')),
    $('div',null,$('div',{class:'sub'},'Downstream'),table(G.downstream,'Readers of this unit.')));
  sec({id:'circuit',wide:true},'Circuit',
    [P('Who writes to the unit and who reads from it, summed over the atlas graphs where it appears'+
       (G.n_atlas_graphs ? ' ('+G.n_atlas_graphs+' graphs)' : '')+'. Weight tiles are tinted by sign and magnitude.')],
    columns);
})();

"""

READING = '<style id="paper-reading-style">\n.pr-box{border:1px solid var(--rule2,#d9dde0);border-radius:8px;padding:14px 18px 12px;margin:0 0 26px;background:#fafbfb;font-size:15px;line-height:1.5}\n.pr-top{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:baseline;cursor:pointer;list-style:none}\n.pr-top::-webkit-details-marker{display:none}\n.pr-top::after{content:\'show\';margin-left:auto;font-size:12px;color:var(--ink3,#6b7178)}\n.pr-box[open] .pr-top::after{content:\'hide\'}\n.pr-k{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3,#6b7178)}\n.pr-g{font-weight:600}.pr-s{font-size:13px;color:var(--ink3,#6b7178)}\n.pr-cols{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:4px 22px}\n.pr-cols h4{margin:6px 0 2px;font-size:13px}.pr-cols ul{margin:0;padding-left:18px}\n.pr-note{font-size:12px;color:var(--ink3,#6b7178)}\n</style>\n<script id="paper-reading-render">\n(() => {\n  window.NEURON_PAPER_REVIEW_LABELS=__REVIEW_LABELS__;\n  const labels=window.NEURON_PAPER_REVIEW_LABELS;\n  const R=B.agent_read;if(!R)return;\n  const el=(t,c,txt)=>{const e=document.createElement(t);if(c)e.className=c;if(txt!=null)e.textContent=txt;return e;};\n  const box=el(\'details\',\'pr-box\'),top=el(\'summary\',\'pr-top\');\n  top.append(el(\'span\',\'pr-k\',\'An agent’s read\'),el(\'span\',\'pr-g\',(labels[R.group]||R.group_name||\'—\')),\n    el(\'span\',\'pr-s\',\'organized \'+R.organized+\'/5 · understood \'+R.understood+\'/5\'));\n  box.append(top,el(\'p\',null,R.one_line));\n  if(R.poles){\n    \n    const text=\'Pole by pole: positive \'+(labels[R.poles.positive?.group]||\'—\')+\n      \', negative \'+(labels[R.poles.negative?.group]||\'—\')+\'.\';\n    box.append(el(\'p\',\'pr-s\',text));\n  }\n  const cols=el(\'div\',\'pr-cols\');\n  const list=(title,items)=>{const d=el(\'div\');d.append(el(\'h4\',null,title));const ul=el(\'ul\');\n    (items||[]).forEach(x=>ul.append(el(\'li\',null,x)));d.append(ul);return d;};\n  cols.append(list(\'Where it branches\',(R.where_it_branches||[]).map(b=>b.branch+\', grown from \'+b.grows_from+(b.path?\': \'+b.path:\'\'))),\n    list(\'What’s concerning\',R.concerning),list(\'What’s fine\',R.fine));\n  box.append(cols,el(\'p\',\'pr-note\',\'Written by an independent DeepSeek high reader from this run’s evidence. The scores are its judgment.\'));\n  const anchor=document.querySelector(\'#head .cfnote\')||document.querySelector(\'#head .lede\')||document.querySelector(\'h1\');\n  if(anchor)anchor.after(box);\n})();\n</script>'.replace('__REVIEW_LABELS__',json.dumps(GROUP_LABELS,ensure_ascii=False))

FULL_READING = '<style id="paper-reading-full-style">\n.prf{max-width:760px;margin:56px auto 40px;padding:26px 0 0;border-top:1px solid var(--rule2,#d9dde0);font-size:16px;line-height:1.6;scroll-margin-top:72px}\n.prf-k{font-size:11px;font-weight:600;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3,#6b7178)}\n.prf h2{font-size:26px;line-height:1.25;margin:6px 0}.prf h3{font-size:15px;margin:26px 0 8px}.prf p{margin:0 0 10px}\n.prf-meta{font-size:14px;color:var(--ink3,#6b7178)}.prf-poles{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:12px 22px}\n.prf-pole{border:1px solid var(--rule2,#d9dde0);border-radius:6px;padding:10px 12px}.prf-pole b{display:block;margin:0 0 4px}\n.prf-item{border-left:3px solid var(--rule2,#d9dde0);padding:2px 0 2px 14px;margin:0 0 14px}.prf-row{font-size:15px;margin:2px 0}.prf-row span{color:var(--ink3,#6b7178);margin-right:8px}\n@media(max-width:820px){.prf{margin:40px 16px}}\n</style><script id="paper-reading-full-render">\n(() => {\n  const labels=window.NEURON_PAPER_REVIEW_LABELS;\n  const R=B.agent_read_full||B.agent_read;if(!R)return;\n  const el=(tag,cls,text)=>{const e=document.createElement(tag);if(cls)e.className=cls;if(text!=null)e.textContent=text;return e;};\n  const section=el(\'section\',\'prf\');section.id=\'agents-read\';\n  section.append(el(\'div\',\'prf-k\',\'An agent’s read, in full\'),el(\'h2\',null,(labels[R.group]||R.group_name||\'—\')),\n    el(\'p\',\'prf-meta\',`organized ${R.organized}/5 · understood ${R.understood}/5`));\n  \n  if(R.poles){\n    section.append(el(\'h3\',null,\'Pole by pole\'));const columns=el(\'div\',\'prf-poles\');\n    Object.entries(R.poles).forEach(([pole,item])=>{const box=el(\'div\',\'prf-pole\');\n      box.append(el(\'b\',null,`${pole===\'positive\'?\'Positive\':\'Negative\'} pole: ${labels[item.group]||item.group}`),\n        el(\'p\',null,item.summary||item.notes||\'—\'));columns.append(box);});section.append(columns);\n  }\n  section.append(el(\'h3\',null,\'What it does\'),el(\'p\',null,R.main_job||R.one_line||\'—\'));\n  section.append(el(\'h3\',null,\'Where it branches\'));\n  const row=(name,text,box)=>{if(text){const p=el(\'div\',\'prf-row\');p.append(el(\'span\',null,name),document.createTextNode(text));box.append(p);}};\n  (R.where_it_branches||[]).forEach(item=>{const box=el(\'div\',\'prf-item\');box.append(el(\'b\',null,item.branch));\n    row(\'Grows from\',item.grows_from,box);row(\'Path\',item.path,box);row(\'Size\',item.size,box);row(\'Evidence\',item.evidence,box);section.append(box);});\n  section.append(el(\'h3\',null,\'What’s left over\'));\n  if(!(R.remainders||[]).length)section.append(el(\'p\',null,\'—\'));\n  (R.remainders||[]).forEach(item=>{const box=el(\'div\',\'prf-item\');box.append(el(\'b\',null,item.what));\n    row(\'Evidence\',item.evidence,box);section.append(box);});\n  [[\'What’s concerning\',R.concerning],[\'What’s fine\',R.fine]].forEach(([title,items])=>{\n    section.append(el(\'h3\',null,title));const list=el(\'ul\');(items||[]).forEach(text=>list.append(el(\'li\',null,text)));section.append(list);});\n  document.getElementById(\'main\').append(section);\n})();\n</script>'

ORDER = """<script id="paper-layout-order">
(() => {
  const main=document.getElementById('main'), nav=document.getElementById('nav');
  const order=['claim','custom','graphs','distribution','token-selectivity','spectrum','exemplars',
    'search','clusters','anatomy','ablation','family','circuit','tests','criticism','interpretation-costs','appendix','agents-read'];
  order.forEach(id=>{
    const section=document.getElementById(id); if(section)main.append(section);
    const link=nav.querySelector('a[href="#'+id+'"]'); if(link)nav.append(link);
  });
})();
</script>"""


def headline(bundle, report, folders=(), data=None):
    """Show each observed pole in its own measured frame and saved context."""
    from pole_selection import select, corpus_strips
    current=copy.deepcopy(bundle.get('headline') or {})
    report=Path(report)
    if data is None:data=read(report/'report.json') if (report/'report.json').exists() else {}
    chosen=select(bundle,report,folders)
    preferred=[(bundle.get('anatomy') or {}).get('strip',{}).get('case_id')]
    strips=corpus_strips(data,preferred)
    panels=[]
    for pole in ['positive','negative']:
        item=chosen.get(pole);ladder=None
        if item:
            figure=item['figure'];conditions={r['condition']:r for r in item['rows']}
            fillers=[]
            for index,row in enumerate(figure.get('results',[])):
                filler={'tag':row.get('tag',row.get('filler','')),'text':row.get('filler',''),
                    'full_text':row.get('text',''),'activation':row['activation'],
                    'peak_token':row.get('peak_token'),'peak_index':row.get('peak_index')}
                condition=conditions.get(row.get('condition',index))
                if condition and condition.get('text')==row.get('text'):
                    trace=condition.get('trace',[])
                    filler.update(tokens=[t['token'] for t in trace],values=[t['product'] for t in trace])
                fillers.append(filler)
            ladder={'template':figure.get('template'),'fillers':fillers,
                    'caption':figure.get('caption'),'source_figure_id':figure['id'],'pole':pole}
        if ladder or pole in strips:panels.append({'pole':pole,'ladder':ladder,'strip':strips.get(pole)})
    if not panels:return None
    # Legacy keys remain readable by callers; the renderer uses both pole panels.
    current.update(poles=panels,ladder=panels[0].get('ladder'))
    if strips:current['strip']=panels[0].get('strip')
    return current


HEADLINE_RENDER = "const headlineFigure = () => { const Hd=B.headline; if(!has(Hd))return null;\n  const wrap=$('div',{class:'headline-poles'});\n  const panels=has(Hd.poles)?Hd.poles:[{ladder:Hd.ladder,strip:Hd.strip}];\n  panels.forEach(panel=>{const row=$('div',{class:'headline'}),L=panel.ladder;\n    if(has(L)&&has(L.fillers)){\n      const F=[...L.fillers].sort((a,b)=>(b.activation||0)-(a.activation||0));\n      const lo=Math.min(0,...F.map(f=>f.activation||0)),hi=Math.max(0,...F.map(f=>f.activation||0));\n      const lines=t=>String(t||'').split(/\\s+/).reduce((out,w)=>{if(!out.length||out[out.length-1].length+w.length+1>28)out.push(w);else out[out.length-1]+=' '+w;return out;},[]); const labels=F.map(f=>lines(f.tag)); const LB=210,W=520,RB=56,rh=24+13*(Math.max(1,...labels.map(x=>x.length))-1),H=rh*F.length+8;\n      const X=v=>LB+(v-lo)/((hi-lo)||1)*(W-LB-RB);\n      const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',class:'ladder',style:'display:block'});\n      const hover=$('div',{class:'hover'}),show=f=>{const ta=has(f.tokens)?zipToks(f.tokens,f.values):null;\n        const pk=ta?(Number.isInteger(f.peak_index)?f.peak_index:argmaxAbs(f.values)):0;\n        hover.replaceChildren($('span',{class:'k'},f.tag),ta?toks(ta,pk,null,(f.activation||0)<0):$('span',{class:'toks'},f.full_text||f.text||''));};\n      svg.append(svgEl('line',{x1:X(0),x2:X(0),y1:2,y2:H-4,stroke:'#c9c9ce'}));\n      F.forEach((f,i)=>{const y=4+i*rh,v=f.activation||0,g=svgEl('g',{class:'row'});\n        g.append(svgEl('rect',{class:'bg',x:0,y,width:W,height:rh,fill:'transparent'}));\n        labels[i].forEach((line,j)=>g.append(svgEl('text',{x:LB-8,y:y+rh/2+4+(j-(labels[i].length-1)/2)*13,'text-anchor':'end','font-size':12,fill:'#1d1d1f'},line)));\n        g.append(svgEl('rect',{x:Math.min(X(0),X(v)),y:y+rh/2-7,width:Math.max(1,Math.abs(X(v)-X(0))),height:14,rx:2,fill:v>=0?POS:NEG}));\n        g.append(svgEl('text',{x:v>=0?X(v)+6:X(0)+6,y:y+rh/2+4,'font-size':12,fill:'#4a4a4f'},fmt(v)));\n        g.onmouseenter=()=>show(f);svg.append(g);});\n      if(F.length)show(panel.pole==='negative'?F[F.length-1]:F[0]);\n      let tmpl=null;if(L.template){tmpl=$('div',{class:'tmpl'});L.template.split(/(\\{[A-Za-z_]+\\})/).forEach(t=>tmpl.append(/^\\{[A-Za-z_]+\\}$/.test(t)?$('span',{class:'slot'},t.slice(1,-1)):t));}\n      row.append($('div',{class:'panel'},panel.pole?$('div',{class:'sub'},panel.pole==='negative'?'Negative-pole test':'Positive-pole test'):null,\n        fig($('div',null,svg,tmpl,hover),L.caption||null)));\n    }\n    const S=panel.strip;\n    if(has(S)&&has(S.tokens)){\n      const vals=S.values||[],mx=Math.max(...vals.map(Math.abs),1e-9),pk=Number.isInteger(S.anchor_index)?S.anchor_index:argmaxAbs(vals);\n      const strip=$('div',{class:'strip'});\n      S.tokens.forEach((t,i)=>{const v=vals[i]||0,h=40*Math.abs(v)/mx;\n        const bar=$('i',{style:v>=0?`bottom:16px;height:${h}px;background:${POS}`:`top:44px;height:${Math.min(h,14)}px;background:${NEG};border-radius:0 0 2px 2px`,title:fmt(v)});\n        strip.append($('span',{class:'stk'},$('span',{class:'bars'},$('i',{class:'base'}),Math.abs(v)/mx>.02?bar:null),\n          $('span',{class:'tk'+(i===pk?' peak'+(v<0?' n':''):''),title:fmt(v)},t)));});\n      row.append($('div',{class:'panel'},fig(strip,S.caption||null)));\n    }\n    if(row.children.length===1)row.classList.add('single');if(row.children.length)wrap.append(row);\n  });\n  return wrap.children.length ? wrap : null; };"

HORIZONTAL_BARS = "const hbars = rows => {\n    const lines=t=>String(t||'').split(/\\s+/).reduce((out,w)=>{if(!out.length||out[out.length-1].length+w.length+1>30)out.push(w);else out[out.length-1]+=' '+w;return out;},[]);\n    const labels=rows.map(r=>lines((r.label||'')+(r.sub?' '+r.sub:''))),vals=rows.map(r=>r.value||0);\n    const lo=Math.min(0,...vals),hi=Math.max(0,...vals),LB=230,W=640,RB=56,rh=24+13*(Math.max(1,...labels.map(x=>x.length))-1),H=rh*rows.length+8;\n    const X=v=>LB+(v-lo)/((hi-lo)||1)*(W-LB-RB),svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',class:'ladder',style:`display:block;max-width:${W}px`});\n    svg.append(svgEl('line',{x1:X(0),x2:X(0),y1:2,y2:H-4,stroke:'#c9c9ce'}));\n    rows.forEach((r,i)=>{const y=4+i*rh,v=r.value||0,g=svgEl('g',{class:'row'});\n      g.append(svgEl('rect',{class:'bg',x:0,y,width:W,height:rh,fill:'transparent'}));\n      labels[i].forEach((line,j)=>g.append(svgEl('text',{x:LB-8,y:y+rh/2+4+(j-(labels[i].length-1)/2)*13,'text-anchor':'end','font-size':12,fill:'#1d1d1f'},line)));\n      g.append(svgEl('rect',{x:Math.min(X(0),X(v)),y:y+rh/2-6,width:Math.max(1,Math.abs(X(v)-X(0))),height:12,rx:2,fill:v>=0?POS:NEG}));\n      g.append(svgEl('text',{x:v>=0?X(v)+6:X(0)+6,y:y+rh/2+4,'font-size':11.5,fill:'#4a4a4f'},fmt(v)));\n      if(r.hover){g.onmousemove=ev=>showTip(ev,r.hover);g.onmouseleave=hideTip;}svg.append(g);\n    });return svg; };"

def anatomy_colours(source):
    # Reference rule: a cluster inherits its first coloured member group's colour.
    # Unknown clusters keep the original grey fallback, independent of product sign.
    source=re.sub(r"const clusterColour = id => \{ if \(SPEC_COLOUR\[id\]\) return SPEC_COLOUR\[id\];",
                  'const clusterColour = id => {',source)
    def scatter(match):
        part=match[0]
        if '// anatomy-cluster-colours-v1' in part:return part
        part=part.replace('clusterColour(', 'anatomyColour(').replace('clusterName(', 'anatomyName(')
        setup="""// anatomy-cluster-colours-v1
  const anatomyClusters=A.clusters || B.clusters || [];
  const groupColours=A.semantic_groups ? {...polePalette(A.semantic_groups),...SPEC_COLOUR} : SPEC_COLOUR;
  const clusterPalette=polePalette(anatomyClusters),anatomyColours={};
  anatomyClusters.forEach(k=>{const member=(k.member_groups||[]).find(id=>groupColours[id]);
    anatomyColours[k.id]=(member ? groupColours[member] : clusterPalette[k.id]) || '#b5b5ba';});
  const anatomyColour=id=>anatomyColours[id] || clusterColour(id);
  const anatomyName=id=>(anatomyClusters.find(k=>k.id===id)||{}).name || clusterName(id);
  """
        part=part.replace('const figs = []; const pts = A.points || [];',setup+'const figs = []; const pts = A.points || [];',1)
        part=part.replace('(B.clusters || []).findIndex(k => k.id === a)', 'anatomyClusters.findIndex(k => k.id === a)')
        part=part.replace('(B.clusters || []).findIndex(k => k.id === b)', 'anatomyClusters.findIndex(k => k.id === b)')
        return part
    return re.sub(r"\(\(\) => \{ const A = B\.anatomy;.*?\}\)\(\);",scatter,source,count=1,flags=re.S)


FAMILY_HEATMAP = r"""
  if (fam.length && Ms.length) {
    let sel=0, start=0;
    const holder=$('div',{style:'overflow-x:auto;max-width:100%;min-width:0'});
    const select=$('select',{class:'sel',style:'max-width:100%'},Ms.map((m,i)=>
      $('option',{value:i},(m.prompt || `sentence ${i+1}`).slice(0,96))));
    const slider=$('input',{type:'range',min:0,step:1,value:0,
      'aria-label':'Family heatmap context window',style:'flex:1;min-width:120px'});
    const position=$('span',{style:'font-variant-numeric:tabular-nums;white-space:nowrap'});
    const controls=$('div',{class:'ctlrow',style:'display:flex;align-items:center;gap:12px;flex-wrap:wrap'},
      'Context window ',slider,position);
    const windowSize=()=>Math.max(8,Math.min(48,Math.floor(
      ((holder.clientWidth || Math.min(window.innerWidth-64,1340))-300)/30)));
    const anchor=m=>{
      const values=(m.readouts || []).find(isSelf)?.values || [];
      return Number.isInteger(m.source_position)
        ? Math.max(0,Math.min((m.tokens || []).length-1,m.source_position-(m.source_start || 0)))
        : argmaxAbs(values);
    };
    const reset=()=>{const m=Ms[sel],n=(m.tokens || []).length,k=windowSize();
      start=Math.max(0,Math.min(Math.max(0,n-k),anchor(m)-Math.floor(k/2)));};
    const drawHeat=()=>{
      holder.replaceChildren();
      const m=Ms[sel],T=m.tokens || [],k=windowSize();
      start=Math.max(0,Math.min(start,Math.max(0,T.length-k)));
      const end=Math.min(T.length,start+k),visible=T.slice(start,end);
      slider.max=Math.max(0,T.length-k);slider.value=start;slider.disabled=T.length<=k;
      position.textContent=T.length ? `Tokens ${start+1}–${end} of ${T.length}` : 'No tokens';
      const L=300,cw=30,rh=26,top=8,botLab=115,W=L+cw*visible.length+8,H=top+rh*fam.length+botLab;
      const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,width:W,height:H,
        style:`display:block;width:${W}px;height:${H}px;max-width:none;flex:none`});
      fam.forEach((u,i)=>{
        const vals=(m.readouts || []).find(r=>r.layer===u.layer && r.neuron===u.neuron)?.values || [];
        // Keep the scale fixed across the complete input while moving the display window.
        const mx=vals.reduce((v,x)=>Math.max(v,Math.abs(x)),1e-9),y=top+i*rh;
        const label=svgEl('text',{x:L-8,y:y+rh/2+4,'text-anchor':'end','font-size':12,
          'font-weight':isSelf(u)?700:400,fill:'#1d1d1f'},
          `${unitId(u)} · ${(isSelf(u)?'this neuron':(u.label || '')).slice(0,30)}`);
        label.append(svgEl('title',null,`${unitId(u)} · ${u.label || ''}`));svg.append(label);
        visible.forEach((t,j)=>{
          const index=start+j,v=vals[index] || 0;
          const cell=svgEl('rect',{x:L+j*cw,y,width:cw-1.5,height:rh-1.5,rx:2,
            fill:hexA(v>=0?POS:NEG,0.04+0.9*Math.abs(v)/mx)});
          cell.append(svgEl('title',null,`${unitId(u)} · token ${index+1} · ${JSON.stringify(t)} · ${fmt(v)}`));
          svg.append(cell);
        });
      });
      visible.forEach((t,j)=>{
        const x=L+j*cw+cw/2+3,y=top+rh*fam.length+8,text=t.trim() || '␣';
        const label=svgEl('text',{x,y,'font-size':12,fill:'#4a4a4f',
          transform:`rotate(-60 ${x} ${y})`,'text-anchor':'end'},
          text.length>22?text.slice(0,21)+'…':text);
        label.append(svgEl('title',null,t));svg.append(label);
      });
      holder.append(svg);
    };
    slider.oninput=()=>{start=+slider.value;drawHeat();};
    select.onchange=()=>{sel=+select.value;reset();drawHeat();};
    reset();drawHeat();requestAnimationFrame(()=>{reset();drawHeat();});
    window.addEventListener('resize',drawHeat);
    figs.push(fig($('div',null,
      Ms.length>1?$('div',{class:'ctlrow'},'Sentence ',select):null,controls,holder),
      'Each row is one unit of the family, normalised to its own largest |activation| on the complete input. This neuron’s row is bold. Move the slider to view the full input.'));
  }
  """

def family_heatmap(source):
    marker='// 5c. Where it sits:'
    if marker not in source:return source
    offset=source.index(marker)
    start=source.index('  if (fam.length && Ms.length)',offset)
    stop=source.index('const up = F.upstream',start)
    return source[:start]+FAMILY_HEATMAP+source[stop:]


COSTS = r"""
<style id="paper-costs-style">
#interpretation-costs{scroll-margin-top:110px}
#interpretation-costs .ic-plots{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:24px;margin:24px 0}
#interpretation-costs .ic-chart{display:block;width:100%;height:196px}
#interpretation-costs .ic-chart text{font-size:12px;fill:var(--ink2,#4a4a4f)}
#interpretation-costs .ic-top{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap}
#interpretation-costs progress{width:100%;height:8px;accent-color:#2679b9;margin-top:12px}
#interpretation-costs .ic-table{width:100%;table-layout:fixed}
#interpretation-costs .ic-table th:first-child{width:48%}
#interpretation-costs .ic-table td{vertical-align:top}
#interpretation-costs .ic-table .ic-num{text-align:right;font-variant-numeric:tabular-nums}
#interpretation-costs .ic-stage{display:flex;gap:10px;align-items:center}
#interpretation-costs .ic-stage > div{min-width:0;overflow-wrap:anywhere}
#interpretation-costs .ic-stage small{display:block;color:var(--ink3,#8b8b94);margin-top:3px}
#interpretation-costs .ic-detail{margin:18px 0 8px}
#interpretation-costs .ic-foot{font-size:13px;color:var(--ink3,#8b8b94)}
@media(max-width:650px){#interpretation-costs .ic-plots{grid-template-columns:1fr;gap:8px}#interpretation-costs .ic-stage{gap:6px}#interpretation-costs .ic-table th:first-child{width:44%}}
</style>
<script id="paper-costs-render">
(() => {
  const C=B.interpretation_costs;
  if(!C?.steps?.length)return;
  let selected=C.completed_steps || C.steps.length;
  const rows=C.steps;
  const money=v=>'$'+Number(v||0).toFixed(4);
  const duration=v=>{const seconds=Math.round(v||0);return seconds?Math.floor(seconds/60)+' min '+String(seconds%60).padStart(2,'0')+' s':'0';};
  const status=$('span'),cumulative=$('span'),bar=$('progress',{max:rows.length,value:selected,'aria-label':'Pipeline stage replay'});
  const plots=$('div',{class:'ic-plots'}),containers=['gpu_seconds','api_usd'].map(key=>{
    const node=$('div');node.dataset.measure=key;plots.append(node);return node;
  });
  const table=$('table',{class:'ic-table'},$('thead',null,$('tr',null,
    $('th',null,'Stage'),$('th',{class:'ic-num'},'GPU · 4-GPU time'),$('th',{class:'ic-num'},'API estimate'))));
  const body=$('tbody');table.append(body);
  const buttons=[];
  rows.forEach(row=>{
    const button=$('button',{class:'more',type:'button','aria-label':'View stage '+row.step},String(row.step));
    button.onclick=()=>select(row.step);buttons.push(button);
    body.append($('tr',null,
      $('td',null,$('div',{class:'ic-stage'},button,$('div',null,row.name,$('small',null,row.resource)))),
      $('td',{class:'ic-num'},duration(row.gpu_seconds)),
      $('td',{class:'ic-num'},money(row.api_usd))));
  });
  const detail=$('p',{class:'ic-detail','aria-live':'polite'});
  const content=$('div',null,$('div',{class:'ic-top'},status,cumulative),bar,plots,table,detail,
    $('p',{class:'ic-foot'},C.basis));
  sec({id:'interpretation-costs',wide:true},'Interpretation Costs',null,content);
  function draw(container,key) {
    const W=container.getBoundingClientRect().width;if(!W)return;
    const H=196,L=62,R=16,T=30,D=42;
    const values=rows.map(row=>key==='gpu_seconds'?row[key]/60:row[key]);
    const high=Math.max(...values,1e-9)*1.28;
    const x=i=>L+6+i*(W-L-R-12)/Math.max(1,rows.length-1);
    const y=v=>H-D-5-v/high*(H-T-D-11);
    const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,class:'ic-chart',role:'img'});
    svg.append(svgEl('title',null,key==='gpu_seconds'?'Four-GPU equivalent runtime by stage':'API cost estimate by stage'));
    svg.append(svgEl('rect',{x:L,y:T,width:Math.max(1,W-L-R),height:H-T-D,fill:'none',stroke:'#d9d9de','stroke-width':1}));
    [0,high/2,high].forEach(value=>{
      svg.append(svgEl('text',{x:L-7,y:y(value)+4,'text-anchor':'end'},key==='gpu_seconds'?value.toFixed(0):value.toFixed(2)));
    });
    const points=values.map((value,i)=>[x(i),y(value)]);
    let path=`M${points[0][0]},${points[0][1]}`;
    points.slice(1).forEach((point,i)=>{const prior=points[i],mid=(prior[0]+point[0])/2;
      path+=` C${mid},${prior[1]} ${mid},${point[1]} ${point[0]},${point[1]}`;});
    const colour=key==='gpu_seconds'?NEG:POS;
    svg.append(svgEl('path',{d:path+` L${x(rows.length-1)},${y(0)} L${x(0)},${y(0)} Z`,fill:hexA(colour,.08)}),
      svgEl('path',{d:path,fill:'none',stroke:colour,'stroke-width':2}));
    points.forEach((point,i)=>{
      svg.append(svgEl('circle',{cx:point[0],cy:point[1],r:i+1===selected?6:3.5,fill:colour}));
      if(W>=410 || i===0 || i===rows.length-1 || i===2)
        svg.append(svgEl('text',{x:point[0],y:H-D+17,'text-anchor':'middle'},String(i+1)));
    });
    const anchor=points[Math.max(0,selected-1)];
    svg.append(svgEl('line',{x1:anchor[0],x2:anchor[0],y1:T,y2:H-D,stroke:'#b5b5ba','stroke-width':1}));
    const peak=values.indexOf(Math.max(...values));
    svg.append(svgEl('text',{x:L,y:16,class:'axis-title','data-axis':'y'},
      key==='gpu_seconds'?'4-GPU time (minutes)':'API estimate (USD)'),
      svgEl('text',{x:(L+W-R)/2,y:H-5,'text-anchor':'middle',class:'axis-title','data-axis':'x'},'Pipeline stage'));
    if(values[peak]>0)svg.append(svgEl('text',{x:x(peak),y:y(values[peak])-11,'text-anchor':peak>=rows.length-2?'end':'middle'},
      key==='gpu_seconds'?duration(rows[peak].gpu_seconds):money(rows[peak].api_usd)));
    container.replaceChildren(svg);
  }
  function select(step) {
    selected=step;
    status.textContent='Stage '+step+' / '+rows.length+' · '+rows[step-1].name;
    cumulative.textContent='Cumulative API '+money(rows.slice(0,step).reduce((sum,row)=>sum+row.api_usd,0));
    bar.value=step;
    detail.textContent=rows[step-1].detail;
    buttons.forEach((button,i)=>button.setAttribute('aria-pressed',i+1===step));
    containers.forEach(container=>draw(container,container.dataset.measure));
  }
  select(selected);
  requestAnimationFrame(()=>select(selected));
  new ResizeObserver(()=>containers.forEach(container=>draw(container,container.dataset.measure))).observe(content);
})();
</script>
"""

def render(source):
    # Saved HTML can still embed the old adapter; keep the reference on/off labels.
    source=re.sub("\\n[ \\t]*anatomy\\.querySelectorAll\\('svg text'\\)\\.forEach\\((node|e)=>\\{\\s*const labels=\\{'gate closed: off':'raw gate < 0','on, positive':'gate > 0, up > 0','on, negative':'gate > 0, up < 0'\\};\\s*if\\(labels\\[\\1\\.textContent\\]\\)\\1\\.textContent=labels\\[\\1\\.textContent\\];\\s*\\}\\);", '', source)
    old_caption="`${(A.n || pts.length).toLocaleString()} points, one per labeled token. ${pct(on.length / Math.max(1, pts.length))} have the gate open; of those, ${pct(onPos.length / Math.max(1, on.length))} have positive up and so a positive activation. Points in the shaded left band are off whatever up says.`"
    legacy_caption="A.selection ? "+chr(96)+"${A.selection.corpus_anchors} selected natural-corpus anchors and ${A.selection.probe_anchors} measured probe anchors. Colours follow semantic cluster membership; unlabelled points are grey. ${pct(on.length / Math.max(1, pts.length))} of these selected points have gate > 0."+chr(96)+" : "+old_caption
    new_caption="A.selection ? "+chr(96)+"${(A.n || pts.length).toLocaleString()} points: ${A.selection.corpus_anchors} selected corpus anchors${A.selection.scan_anchors ? ', '+A.selection.scan_anchors+' amplitude-stratified corpus scan anchors' : ''}, and ${A.selection.probe_anchors} measured probe anchors. Colours follow semantic cluster membership; unlabelled points are grey. ${pct(on.length / Math.max(1, pts.length))} of these selected points have gate > 0."+chr(96)+" : "+old_caption
    source=source.replace(legacy_caption,new_caption)
    if 'amplitude-stratified corpus scan anchors' not in source:source=source.replace(old_caption,new_caption)
    source=re.sub(r'const hbars = rows => \{.*?return svg; \};',lambda m:HORIZONTAL_BARS,source,count=1,flags=re.S)
    source=source.replace('Existing semantic labels are retained; unlabelled probes are grey.','Colours follow semantic cluster membership; unlabelled points are grey.')
    source=anatomy_colours(source)
    source=family_heatmap(source)
    # These are layout changes to the shared renderer, never to an individual result.
    source=re.sub(r'const headlineFigure = \(\) => \{.*?return wrap.children.length \? wrap : null; \};',
                  lambda m:HEADLINE_RENDER,source,count=1,flags=re.S)
    source=source.replace("A.headline.slice(0, 2).map(s => $('div', null, bars3(s)))",
        "A.headline.slice(0, 2).map(s => $('div', null, s.pole ? $('div',{class:'sub'},s.pole==='negative'?'Negative response':'Positive response') : null, bars3(s)))")
    source=source.replace('Two measured replacements in the same frame; gate, value and product are shown for every token.',
        'Representative measured replacements from the positive and negative tests; each sentence retains its own frame. Gate, value and product are shown for every token.')
    source=source.replace('Effect of zeroing the unit on each graph’s target, as measured.',
        'Zeroing this neuron at the graph anchor. Effect is the logit change for the reference token named in each row.')
    # Replace both the old median-based adapter and the original short declarations.
    source = re.sub(r'// Shared display colours\..*?(?=// Pole-and-order palette)',
                    lambda m:COLOURS, source, count=1, flags=re.S)
    source = re.sub(r'const poleOf = g => .*?;\nconst isOther = g => .*?;\n',
                    lambda m:COLOURS.split('\n',1)[1], source, count=1, flags=re.S)
    source = source.replace("clusters:['Clusters',['Clusters','Routes','Route factorization']]",
                            "clusters:['Clusters',['Clusters']]")
    # The publication view uses the main distribution, without a second scan panel.
    source = re.sub(r'  // Larger scan:.*?(?=  sec\(\{ id: \x27distribution\x27)',
                    '', source, count=1, flags=re.S)
    source = source.replace("), tiles, big); })();", "), tiles); })();", 1)
    source = re.sub(r'// 6\. Circuit tables:.*?(?=// 7\. Tests\.)',lambda m:CIRCUIT,
                    source,count=1,flags=re.S)
    source = source.replace("circuit:['Circuit',['Upstream','Downstream','Logits']]",
                            "circuit:['Circuit',['Upstream','Downstream']]")
    source = source.replace(
        "const pk = ta ? (f.peak_token ? Math.max(0, f.tokens.lastIndexOf(f.peak_token)) : argmaxAbs(f.values)) : 0;",
        "const pk = ta ? (Number.isInteger(f.peak_index) ? f.peak_index : (f.peak_token ? Math.max(0,f.tokens.lastIndexOf(f.peak_token)) : argmaxAbs(f.values))) : 0;")
    # Keep all three test slots, including when only one measured result exists.
    source = source.replace("const T = B.tests; if (!has(T)) return; const figs = [];",
                            "const T = B.tests || {}; const figs = [];")
    source = source.replace(
        "if (bm || bn) figs.push(fig($('div', null, bm, bn), 'The explanation is used to write text without looking at the corpus. Achieved is the model’s measured activation on that text.'));",
        "const missing = name => $('div',null,$('div',{class:'sub'},name),$('span',{class:'tile'},'—'));\n"
        "  figs.push(fig($('div',null,bm || missing('Blind maximum'),bn || missing('Blind minimum')),\n"
        "    bm || bn ? 'The explanation is used to write text without looking at the corpus. Achieved is the model’s measured activation on that text.' : null));")
    if "missing('Route prediction')" not in source:
        source = source.replace("  if (has(T.steering))",
                                "  if (!has(rp)) figs.push(fig(missing('Route prediction')));\n  if (has(T.steering))",1)
    # Remove the legacy end-of-page fallback for the same circuit slots.
    source = re.sub(r"  if\(B\.circuit\?\.logits && !B\.circuit\?\.upstream\?\.length && !B\.circuit\?\.downstream\?\.length\)missingTiles\('circuit',\['Upstream','Downstream'\]\);\n?",'',source)
    source = re.sub(r'<style id="paper-reading-style">.*?</style>','',source,flags=re.S)
    source = re.sub(r'<script id="paper-reading-render">.*?</script>','',source,flags=re.S)
    source = re.sub(r'<script id="paper-layout-order">.*?</script>','',source,flags=re.S)
    source = re.sub(r'<style id="paper-costs-style">.*?</style>','',source,flags=re.S)
    source = re.sub(r'<script id="paper-costs-render">.*?</script>','',source,flags=re.S)
    source = re.sub(r'<!-- paper-review-full:start -->.*?<!-- paper-review-full:end -->','',source,flags=re.S)
    source = re.sub(r'<style id="paper-reading-full-style">.*?</style>','',source,flags=re.S)
    source = re.sub(r'<script id="paper-reading-full-render">.*?</script>','',source,flags=re.S)
    source = source.replace('Short minimal pairs read the same way.','Representative measured replacements from the positive and negative tests; each sentence retains its own frame. Gate, value and product are shown for every token.')
    if '</body>' in source:
        head,tail=source.split('</body>',1)
        return head.rstrip()+'\n'+READING+'\n'+FULL_READING+'\n'+COSTS+'\n'+ORDER+'\n</body>'+tail
    return source.rstrip()+'\n'+READING+'\n'+FULL_READING+'\n'+COSTS+'\n'+ORDER
