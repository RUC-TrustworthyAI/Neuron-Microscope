(() => {
'use strict';
const {language,t,updateLanguageLinks}=window.NeuronDemoI18n;
const $=(s,p=document)=>p.querySelector(s);
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=x=>Number(x).toLocaleString(language==='zh'?'zh-CN':'en-US');
const pct=x=>(100*x).toFixed(1)+'%';
const num=x=>x==null?'—':Number(x).toLocaleString('en-US',{maximumFractionDigits:3});
function el(tag,cls,text){const n=document.createElement(tag);if(cls)n.className=cls;if(text!=null)n.textContent=text;return n;}
function p(host,text,cls=''){host.append(el('p',cls,text));}
function button(text,fn,cls='quiet-button'){const b=el('button',cls,text);b.type='button';b.onclick=fn;return b;}
function link(text,url){const a=el('a','',text);a.href=url;a.target='_blank';a.rel='noopener';return a;}
const cache=new Map();async function get(path){if(!cache.has(path))cache.set(path,fetch(path+'?v=reasoning-mix-v2').then(r=>{if(!r.ok)throw Error(t('读取记录失败：','Could not load record: ')+path);return r.json();}));return cache.get(path);}
let data,current=0,activeTab='overview',revision=0,experiment='search',coverSnapshot='final';
const steps=[
 {id:'scan',nav:t('准备证据','Prepare evidence'),role:t('固定程序','Python'),title:t('从100M激活中整理解释证据','Prepare evidence from 100M activations'),summary:t('程序按激活区间均分500个位置，并整理极值、Exemplars、分位样本和token统计。','Code allocates 500 positions equally across activation bins and gathers extrema, exemplars, quantile cases and token statistics.')},
 {id:'initial',nav:t('提出初步解释','Initial explanation'),role:'LLM',title:t('寻找具体概念和表达模式','Identify specific concepts and expression patterns'),summary:t('解释模型结合实测激活与上下文，提出总体解释及正负极条件。','The explanation model reads measured activations and contexts, then proposes an overall account and conditions for each pole.')},
 {id:'loops',nav:t('盲标与两次修订','Blind labels and refinement'),role:t('LLM＋固定程序','LLM + Python'),wide:true,title:t('对照三轮盲标，查看解释如何变化','Compare three rounds of predictions and explanations'),summary:t('每轮按100点一批、5路并行盲标。修订模型看见实测证据和明显失误，连续修改两次。','Each labeling pass uses five parallel batches of 100 points. The refinement model sees measured evidence and focused prediction errors, then updates the explanation twice.')},
 {id:'freeze',nav:t('选择并冻结解释','Select and freeze'),role:t('固定程序','Python'),wide:true,title:t('选择解释，并保留与它对应的盲标结果','Select an explanation together with its predictions'),summary:t('先找方向准确率严格大于90%的版本，再从中选择覆盖率最高者。若没有合格版本，则选择方向准确率最高者。','Among versions with directional accuracy strictly above 90%, select the one with the highest coverage. If none qualify, select the most accurate version.')},
 {id:'experiments',nav:t('构造文本与测量','Construct texts and measure'),role:t('LLM＋GPU测量','LLM + GPU'),wide:true,title:t('用新文本检验冻结的解释','Test the frozen explanation on new inputs'),summary:t('Search逐轮接收测量反馈；受控测试改变一个因素；Blind独立根据冻结解释写作。','Search uses measured feedback, controlled tests vary one factor, and an independent blind writer uses the frozen explanation.')},
 {id:'report',nav:t('归因、评阅与报告','Attribution and report'),role:t('GPU＋LLM＋固定程序','GPU + LLM + Python'),title:t('汇合测量证据，生成交互报告','Assemble measurements into an interactive report'),summary:t('程序测量归因、Family和干预；LLM分析证据并独立评阅，随后由固定代码生成HTML。','Code measures attribution, family responses and interventions. LLM calls analyze and independently assess the evidence, then code renders the HTML.')}
];
const tabs={overview:t('过程讲解','Overview'),prompt:t('实际Prompt','Actual prompts'),actions:t('程序步骤','Program steps'),output:t('阶段产物','Outputs')};
function roundName(r){return r===0?t('初步解释','Initial explanation'):t('第','Refinement ')+r+(language==='zh'?'次修订':'');}
function selected(){return data.rounds.find(r=>r.selected);}
function modal(title,kicker='',cls='text-mode'){const dialog=$('#reading-dialog');$('#reading-title').textContent=title;$('#reading-kicker').textContent=kicker;$('#reading-actions').replaceChildren();const body=$('#reading-content');body.className='reading-content '+cls;body.replaceChildren();if(!dialog.open)dialog.showModal();return body;}
function textModal(title,value){const body=modal(title,t('保存的记录','Saved record'));body.append(el('pre','inspector-text',typeof value==='string'?value:JSON.stringify(value,null,2)));}
function openReport(kind='final',hash=''){const url='reports/'+kind+'.html'+(hash?'#'+hash:'');const body=modal(t('交互式神经元报告','Interactive neuron report'),'Qwen3-32B · Reasoning Mix 100M','iframe-mode');const frame=el('iframe');frame.title='Recorded report';frame.src=url;body.append(frame);$('#reading-actions').append(link(t('单独打开 ↗','Open separately ↗'),url));}
function code(host,title,value){const text=typeof value==='string'?value:JSON.stringify(value,null,2);const wrap=el('div','code-card'),bar=el('div','code-title');bar.append(el('span','',title),button(t('完整阅读','Read in full'),()=>textModal(title,text)));wrap.append(bar,el('pre','',text.length>3600?text.slice(0,3600)+'\n…':text));host.append(wrap);return wrap;}
function details(host,title,render,open=false){const d=el('details','simple');d.append(el('summary','',title));let made=false;d.ontoggle=()=>{if(d.open&&!made){made=true;render(d);}};host.append(d);if(open){made=true;render(d);d.open=true;}return d;}
function table(host,headers,rows,cls=''){const wrap=el('div','table-scroll'),tb=el('table','walk-table '+cls),head=el('tr');headers.forEach(x=>head.append(el('th','',x)));const thead=el('thead');thead.append(head);tb.append(thead);const body=el('tbody');for(const row of rows){const tr=el('tr');for(const value of row){const td=el('td');if(value instanceof Node)td.append(value);else td.textContent=value??'—';tr.append(td);}body.append(tr);}tb.append(body);wrap.append(tb);host.append(wrap);return tb;}
function story(host,rows){for(const [label,body] of rows){const block=el('div','story-block');block.append(el('div','story-label',label),el('div','story-text',body));host.append(block);}}
function explanation(host,round,compact=false){const raw=round.explanation;host.append(el('h3','explanation-title',raw.title));p(host,raw.lede,'explanation-lede');for(const pole of ['positive','negative']){const list=round.rules.filter(r=>r.predicted_pole===pole);const box=el('div','pole-list '+pole);box.append(el('h4','',pole==='positive'?t('正极条件','Positive conditions'):t('负极条件','Negative conditions')));if(!list.length)p(box,t('该方向暂无有证据支持的条件。','No condition is supported for this direction.'));for(const rule of list){const d=el('details','rule');const summary=el('summary');summary.append(el('i','swatch '+pole),el('span','',rule.name));d.append(summary,el('p','',rule.description));box.append(d);}host.append(box);}if(!compact)details(host,t('竞争假设与改变判断的证据','Competing hypotheses and what would change the account'),d=>code(d,t('保存的解释输出','Saved explanation'),raw));}
const warm=['#e34509','#ec6538','#ee8a66','#f0ae97','#f3cbbb'],cool=['#2377b7','#5292c5','#81add0','#b1ccdf','#d2e0ea'];
function palette(round){const colors={};for(const pole of ['positive','negative'])round.rules.filter(r=>r.predicted_pole===pole).forEach((r,i)=>colors[r.id]=(pole==='positive'?warm:cool)[i%5]);colors['no-visible-cue']='#939396';return colors;}
function svgEl(tag,attrs={},text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text!=null)n.textContent=text;return n;}
function spectrum(round,mode='counts',onPick=null){const W=740,H=325,L=48,R=12,T=22,B=66,base=H-B,plotH=base-T,edges=data.spectrum.edges,hist=data.spectrum.corpus_hist,nb=hist.length,band=(W-L-R)/nb,colors=palette(round);const maxlog=Math.log10(Math.max(...hist,10));const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,role:'img','aria-label':t('激活频谱，第','Activation spectrum, version ')+round.round,class:'spectrum-chart'});svg.append(svgEl('title',{},t('柱高为全语料计数；颜色按本轮已标注样本的组别占比分配。','Bar heights are full-corpus counts; colors divide the height by labeled group shares.')));const ticks=mode==='counts'?Array.from({length:Math.floor(maxlog)+1},(_,k)=>10**k):[0,.25,.5,.75,1];for(const v of ticks){const y=base-(mode==='counts'?Math.log10(v)/maxlog:v)*plotH;svg.append(svgEl('line',{x1:L,x2:W-R,y1:y,y2:y,stroke:'#e9e9eb'}));svg.append(svgEl('text',{x:L-7,y:y+4,'text-anchor':'end','font-size':11,fill:'#8d8d91'},mode==='counts'?(v>=1e6?v/1e6+'M':v>=1e3?v/1e3+'k':v):Math.round(v*100)+'%'));}for(let i=0;i<nb;i++){const points=round.points.filter(p=>p.bin===i),height=(mode==='counts'?Math.log10(Math.max(1,hist[i]))/maxlog:1)*plotH;let cum=0;for(const rule of round.rules){const n=points.filter(p=>p.group===rule.id).length;if(!n)continue;const h=n/points.length*height;const rect=svgEl('rect',{x:L+i*band+2,y:base-cum-h,width:band-4,height:Math.max(0,h-1),rx:1,fill:colors[rule.id]||'#939396',class:'chart-segment'});rect.append(svgEl('title',{},`${rule.name}\n${edges[i]}…${edges[i+1]} · ${n}/${points.length}`));if(onPick)rect.onclick=()=>onPick(rule.id,i);svg.append(rect);cum+=h;}const x=L+(i+.5)*band;svg.append(svgEl('text',{x,y:base+13,transform:`rotate(-35 ${x} ${base+13})`,'text-anchor':'end','font-size':10,fill:'#666'},`${edges[i]}…${edges[i+1]}`));svg.append(svgEl('text',{x,y:H-5,'text-anchor':'middle','font-size':10,fill:'#999'},'n '+points.length));}return svg;}
function pointCard(point,group){const row=el('article','point-card');row.append(el('span','activation '+(point.activation<0?'negative':'positive'),num(point.activation)));const body=el('div');const prefix=point.observed_prefix||'',token=point.anchor?.token||'';const text=el('div','sample-text');if(token&&prefix.endsWith(token)){text.append(document.createTextNode(prefix.slice(-420,-token.length)));text.append(el('mark',point.activation<0?'negative':'positive',token));}else text.append(document.createTextNode(prefix.slice(-420)));if(point.nearby_continuation)text.append(el('span','continuation',point.nearby_continuation));body.append(text,el('small','',group||point.reason||''));row.append(body);return row;}
function roundModal(round,group=null,bin=null){const body=modal(roundName(round.round),t('解释与本轮盲标','Explanation and blind predictions'),'round-mode');const plot=el('div','large-plot');plot.append(spectrum(round,'counts',(g,b)=>roundModal(round,g,b)));body.append(plot);const filters=el('div','round-filters');for(const rule of round.rules){const b=button(rule.name,()=>roundModal(round,group===rule.id?null:rule.id,bin),'legend-button');b.style.setProperty('--swatch',palette(round)[rule.id]);b.classList.toggle('active',rule.id===group);filters.append(b);}body.append(filters);if(group||bin!=null){body.append(button(t('清除筛选','Clear filters'),()=>roundModal(round)));const points=round.points.filter(p=>(!group||p.group===group)&&(bin==null||p.bin===bin)).sort((a,b)=>Math.abs(b.activation)-Math.abs(a.activation));p(body,t('已标注位置：','Labeled positions: ')+points.length);for(const point of points.slice(0,20))body.append(pointCard(point,round.rules.find(r=>r.id===point.group)?.name));}else explanation(body,round);}
function roundCards(host){const toolbar=el('div','plot-toolbar');let mode='counts';const charts=[];for(const [key,label] of [['counts',t('全语料计数，log','Corpus counts, log')],['share',t('区间内占比','Share per bin')]]){const b=button(label,()=>{mode=key;toolbar.querySelectorAll('button').forEach(n=>n.classList.toggle('active',n===b));charts.forEach(([target,round])=>target.replaceChildren(spectrum(round,mode,(g,i)=>roundModal(round,g,i))));},'seg-button');b.classList.toggle('active',key===mode);toolbar.append(b);}toolbar.append(el('span','small',t('点击色块查看样本；点击标题展开本轮。','Click a segment to inspect samples; open a title to read the full version.')));host.append(toolbar);const grid=el('div','round-grid');for(const round of data.rounds){const card=el('article','round-card'+(round.selected?' chosen':''));const h=el('div','round-heading');h.append(button(roundName(round.round),()=>roundModal(round),'round-title'));if(round.selected)h.append(el('span','selected-tag',t('最终选用','Selected')));card.append(h,el('p','round-metrics',t('已标注','Labeled ')+`${round.summary.labelled}/500`),el('p','round-metrics',t('覆盖率 ','Coverage ')+pct(round.summary.directional_coverage)+' · '+t('方向准确率 ','Accuracy ')+pct(round.summary.directional_accuracy)));const plot=el('div','round-plot');plot.append(spectrum(round,mode,(g,i)=>roundModal(round,g,i)));charts.push([plot,round]);card.append(plot);explanation(card,round,true);grid.append(card);}host.append(grid);}
function selectionTable(host){const headers=[t('版本','Version'),t('已标注','Labeled'),t('方向准确率','Directional accuracy'),t('覆盖率','Coverage'),t('选择结果','Selection')];table(host,headers,data.rounds.map(r=>[roundName(r.round),r.summary.labelled+'/500',pct(r.summary.directional_accuracy),pct(r.summary.directional_coverage),r.selected?t('最终选用','Selected'):'—']),'selection-table');}
function measuredCard(row){const article=el('article','measured-card'),head=el('div','measured-heading');head.append(el('b','',row.label||row.id),el('span','activation '+(row.peak.product<0?'negative':'positive'),num(row.peak.product)));article.append(head);const text=el('div','sample-text');const offset=row.peak.offset;if(offset){text.append(document.createTextNode(row.text.slice(0,offset[0])),el('mark',row.peak.product<0?'negative':'positive',row.text.slice(...offset)),document.createTextNode(row.text.slice(offset[1])));}else text.textContent=row.text;article.append(text);return article;}
function controlledFigure(host,figure){host.append(el('h3','',figure.title));if(figure.rationale)p(host,figure.rationale);const vals=figure.results.map(r=>r.activation);const lo=Math.min(0,...vals),hi=Math.max(0,...vals),span=hi-lo||1,W=860,L=210,R=60,H=figure.results.length*36+24,zero=L+(-lo/span)*(W-L-R);const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,class:'control-chart',role:'img','aria-label':figure.title});svg.append(svgEl('line',{x1:zero,x2:zero,y1:0,y2:H,stroke:'#c5c5c8'}));figure.results.forEach((r,i)=>{const y=i*36+5,x=L+(r.activation-lo)/span*(W-L-R);svg.append(svgEl('text',{x:L-12,y:y+21,'text-anchor':'end','font-size':17,fill:'#333'},r.tag));const bar=svgEl('rect',{x:Math.min(x,zero),y:y+4,width:Math.max(.7,Math.abs(x-zero)),height:24,rx:3,fill:r.activation<0?'#2377b7':'#e34509',class:'chart-segment'});bar.append(svgEl('title',{},r.text||r.tag));bar.onclick=()=>textModal(r.tag,{text:r.text,activation:r.activation});svg.append(bar,svgEl('text',{x:r.activation<0?x-8:x+8,y:y+22,'text-anchor':r.activation<0?'end':'start','font-size':17,fill:'#666'},num(r.activation)));});host.append(svg);const template=figure.template||'';if(template)p(host,template,'template-text');if(figure.caption)p(host,figure.caption,'small');}
function experimentView(host){const sub=el('div','experiment-tabs');for(const [key,title]of [['search','Search'],['head',t('受控测试与头图','Controlled tests and headline')],['blind','Blind']])sub.append(button(title,()=>{experiment=key;renderTab();},key===experiment?'active':''));host.append(sub);if(experiment==='search'){story(host,[[t('输入','Input'),t('冻结解释、自然语料证据；后两轮还接收上一轮的实测响应。','Frozen explanation and natural evidence; later rounds also receive the previous round’s measurements.')],[t('目标与测量','Objective and measurement'),t('尽可能提高正极响应、降低负极响应。每篇文本测全文最大值或最小值，共进行3轮。','Maximize positive activation and make negative activation as negative as possible. Score full-input maxima or minima across three rounds.')]]);table(host,[t('轮次','Round'),t('文本数','Texts'),t('最高激活','Maximum'),t('最低激活','Minimum')],data.search_progress.map(r=>[r.round,r.candidates,num(r.maximum),num(r.minimum)]));for(const sign of [1,-1]){host.append(el('h3','section-title',sign>0?t('最强正极文本','Strongest positive inputs'):t('最强负极文本','Strongest negative inputs')));data.measurements.filter(r=>r.id.startsWith('search-')&&r.peak.product*sign>0).sort((a,b)=>sign*(b.peak.product-a.peak.product)).slice(0,3).forEach(r=>host.append(measuredCard(r)));}}else if(experiment==='head'){story(host,[[t('设计','Design'),t('基于冻结解释与实测样本，设计4个测试，每个测试10个条件。在共同框架下改变一个因素。','Design four tests with ten conditions each from the frozen explanation and measured examples. Vary one factor within each shared frame.')],[t('测量与改进','Measure and improve'),t('统一测替换片段或其后的共同片段，保留最大绝对值响应的符号。测量后改进一次实验设计；总体解释保持冻结。','Score the changed span or a common suffix, retaining the sign of the largest absolute response. Refine the experimental design once after measurement; the overall explanation stays frozen.')]]);const select=el('select','doc-select');data.figures.forEach(f=>{const o=el('option','',f.title+(f.id===data.headline?t(' · 头图',' · Headline'):''));o.value=f.id;select.append(o);});select.value=data.headline;const body=el('div');const draw=()=>{body.replaceChildren();controlledFigure(body,data.figures.find(f=>f.id===select.value));};select.onchange=draw;host.append(select,body);draw();details(host,t('查看实验改进前后的实测结果','Compare the original and refined measurements'),d=>{for(const [title,list] of [[t('初版测试','Original tests'),data.controls_initial],[t('改进后的测试','Refined tests'),data.controls_revised]])details(d,title,x=>list.forEach(f=>controlledFigure(x,f)));});}else{story(host,[[t('输入','Input'),t('仅提供冻结解释和对应条件。写作模型看不到自然语料样本、Search结果或测量反馈。','Only the frozen explanation and its conditions. The writer receives no natural examples, search results or measurement feedback.')],[t('生成与测量','Write and measure'),t('有解释支持的方向各写6篇文本。正极测每篇全文最大值，负极测全文最小值，再取两端最强者。写作可以与Search并行。','Write six texts for each supported direction. Measure full-input maxima for positive texts and minima for negative texts, then retain the strongest result at each pole. Writing can run in parallel with search.')]]);table(host,[t('来源','Source'),t('最高激活','Maximum'),t('最低激活','Minimum')],[[t('自然语料','Natural corpus'),num(data.corpus.max),num(data.corpus.min)],['Blind',num(data.blind.blind_max?.achieved),num(data.blind.blind_min?.achieved)]]);const list=data.measurements.filter(r=>r.id.startsWith('blind-')).sort((a,b)=>Math.abs(b.peak.product)-Math.abs(a.peak.product));for(const row of list)host.append(measuredCard(row));}}
const blindLabelRounds={'initial-labels':0,'refined-labels':1,'refined-labels-2':2};
const callGroups={
  initial:['initial'],
  loops:['initial-labels','refinement','refined-labels','refinement-2','refined-labels-2'],
  experiments:['search','blind-writer','control-plan','control-refinement'],
  report:['route-descriptions','route-predictions','neuron-labels','report-analysis','final-review'],
  followup:['followup'],
  synthesis:['report-synthesis']
};
function callGroup(id){
  const [group,task]=id.split('/',2);
  return group==='followup'&&task==='report-synthesis'?'report-synthesis':group;
}
function synthesisPart(id){
  return id.replace(/^followup\//,'').split('/').slice(1);
}
function relevantCalls(step){
  const groups=callGroups[step]||[];
  const calls=groups.flatMap(group=>data.calls.filter(call=>callGroup(call.id)===group));
  if(step==='synthesis'){
    const order={sections:0,findings:1,overview:2,'claim-overview':2};
    const rank=call=>order[synthesisPart(call.id)[0]]??-1;
    return calls.sort((a,b)=>rank(a)-rank(b));
  }
  if(step!=='followup')return calls;
  // Within each round, show the plan before its investigations and design before assessment.
  const order=({id})=>id.replace(/\/round-(\d+)\//,'/r$1-i0/').replace(/\/specification$/,'/0');
  return calls.sort((a,b)=>order(a).localeCompare(order(b),undefined,{numeric:true}));
}
const findingTitles={pattern:t('核心响应','Core response'),strength:t('强弱响应','Strong and weak responses'),
  mechanism:t('机制和输出效应','Mechanism and output effects'),open_questions:t('尚待回答的问题','Open questions'),
  main_response:t('主要响应','Main response'),negative_pole:t('负极响应','Negative pole'),
  what_would_change_my_mind:t('什么证据会改变判断','What would change my mind')};
const sectionTitles={ablation:t('点消融','Ablations'),anatomy:'Anatomy',circuit:t('归因回路','Circuit'),
  criticism:t('质疑与追加调查','Criticisms and follow-ups'),custom:t('受控实验','Controlled tests'),family:'Family',
  search:'Search',spectrum:t('激活频谱','Activation spectrum'),tests:t('实验检验','Tests')};
function callName(call){
  const [group,identity,operation]=call.id.split('/');
  if(Object.hasOwn(blindLabelRounds,group)){
    const count=call.input_count==null?'':' · '+call.input_count+t('点',' points');
    return roundName(blindLabelRounds[group])+' · '+t('盲标批次','Blind-label batch')+count;
  }
  const names={
    initial:roundName(0),refinement:roundName(1),'refinement-2':roundName(2),
    'control-plan':t('受控测试设计','Controlled-test design'),
    'control-refinement':t('受控测试改进','Controlled-test refinement'),
    'blind-writer':t('Blind独立写作','Independent blind writer'),
    'route-descriptions':t('路径描述','Route descriptions'),
    'route-predictions':t('盲测路径预测','Blind route prediction')+' · '+identity,
    'neuron-labels':t('相关神经元命名','Related neuron labeling')+' · '+operation,
    'report-analysis':t('报告分析','Report analysis'),
    'final-review':t('独立评阅','Independent review'),
    'report-synthesis':t('证据综合','Evidence synthesis')
  };
  if(group==='search')return 'Search · '+identity.split('-').pop();
  if(callGroup(call.id)==='report-synthesis'){
    const [part,field]=synthesisPart(call.id);
    if(part==='sections')return t('章节摘要','Section summary')+' · '+(sectionTitles[field]||field);
    if(part==='findings')return t('综合结论','Finding')+' · '+(findingTitles[field]||field);
    if(part==='overview'||part==='claim-overview')return field==='title'?t('报告标题','Report title'):t('报告导语','Report lede');
    return names['report-synthesis'];
  }
  if(group==='followup'){
    const phases={plan:t('规划','Planning'),specification:t('实验设计','Test design'),
      assessment:t('结果判定','Assessment')};
    return (phases[operation]||operation)+' · '+identity;
  }
  return names[group]||call.id;
}
async function promptView(host,step,token){
  const calls=relevantCalls(step);
  if(!calls.length){
    if(step==='scan'||step==='freeze'){
      p(host,t('这一步由固定代码执行。','This step runs in fixed code.'));
      code(host,t('程序输入','Program input'),step==='scan'?data.preparation:{selection_rule:'accuracy > 0.9 → maximum coverage; otherwise maximum accuracy',selected_round:data.selected_round});
    }else p(host,t('这一步没有保存的模型调用。','No model calls were saved for this step.'));
    return;
  }
  p(host,t('任务提示保留原文。输入展示实际节选，输出展示本次保存的结果。','Task instructions are shown verbatim. Inputs show recorded excerpts and outputs show the saved results.'));
  const select=el('select','doc-select');
  calls.forEach((call,i)=>{
    const option=el('option','',`${i+1}. ${callName(call)}`);
    option.value=i;
    select.append(option);
  });
  const body=el('div');
  host.append(select,body);
  const draw=async()=>{
    const index=select.value,call=await get(calls[Number(index)].file);
    if(token!==revision||index!==select.value)return;
    body.replaceChildren();
    const plain=call.response_format==='text';
    code(body,plain?t('完整任务提示','Complete task instructions'):
      t('完整任务提示与JSON schema','Complete task instructions and JSON schema'),call.prompt);
    details(body,t('模型看见的输入节选','Input excerpts shown to the model'),d=>{
      p(d,t('部分长列表展示前3项，total为原列表长度。','Long lists show their first three entries; total records the original list length.'),'small');
      code(d,t('输入节选','Input excerpt'),call.input_excerpt);
    });
    if(call.correction_messages?.length)details(body,t('格式纠正','Format correction'),d=>{
      for(const message of call.correction_messages){
        const title=message.role==='assistant'?t('模型的上一次输出','Previous model response'):t('格式纠正请求','Format correction request');
        code(d,title,message.content);
      }
    });
    details(body,plain?t('通过校验的文本','Validated text'):
      t('模型的完整结构化输出','Complete structured output'),d=>code(d,t('输出','Output'),plain?call.output.text:call.output));
  };
  select.onchange=draw;
  await draw();
}
function reportExtras(host,showOutputs){
  const route=data.route_prediction;
  if(route){
    const title=t('路径预测','Route prediction');
    if(showOutputs)details(host,title,d=>code(d,title,route));
    else{
      host.append(el('h3','section-title',title));
      if(route.status==='complete'){
        table(host,[t('已分配路径的测试位置','Assigned test positions'),t('准确率','Accuracy'),
          t('训练集多数路径基线','Training-majority baseline')],[[route.n,pct(route.accuracy),pct(route.baseline)]]);
      }else p(host,route.reason);
      host.append(link(t('查看报告中的路径预测 ↗','View route predictions in the report ↗'),'reports/final.html#tests'));
    }
  }
  if(data.neuron_labels){
    const title=t('相关神经元命名','Related neuron labels');
    if(showOutputs)details(host,title,d=>code(d,title,data.neuron_labels));
    else{
      const labels=Object.values(data.neuron_labels.labels);
      const labelled=labels.filter(row=>row.status==='labelled').length;
      host.append(el('h3','section-title',title));
      table(host,[t('已命名','Labelled'),t('证据不足','Insufficient evidence')],
        [[labelled,labels.length-labelled]]);
      host.append(link(t('查看报告中的神经元命名 ↗','View neuron labels in the report ↗'),'reports/final.html#family'));
    }
  }
}
function followupView(host){
  const {original_criticism,rounds,stop_reason}=data.followup;
  if(original_criticism?.length)details(host,t('报告中的原始疑点','Original criticisms'),d=>{
    const list=el('ul','review-list');
    original_criticism.forEach(row=>list.append(el('li','',row.attack)));
    d.append(list);
  });
  const verdicts={survived:t('通过检验','Survived'),narrowed:t('缩小适用范围','Narrowed'),
    rejected:t('不成立','Rejected'),inconclusive:t('证据不足以判定','Inconclusive')};
  for(const round of rounds){
    host.append(el('h3','section-title',t(`第${round.round}轮`,`Round ${round.round}`)));
    if(round.plan?.reason)p(host,round.plan.reason);
    for(const item of round.investigations){
      const result=item.assessment;
      details(host,item.question.question,d=>{
        story(d,[
          [t('用于区分假设的观测','Distinguishing observation'),item.question.distinguishing_observation],
          [t('观察结果','Observed result'),result?.result],
          [t('判定','Verdict'),item.status==='failed'?t('未完成','Incomplete'):verdicts[result?.verdict]],
          [t('修订建议','Suggested revision'),result?.suggested_revision]
        ].filter(([,value])=>value));
        if(item.evidence)details(d,t('保存的观测记录','Recorded observations'),record=>
          code(record,t('观测证据','Observed evidence'),item.evidence));
      });
    }
  }
  const stops={round_limit:t('达到调查轮次上限','Round limit reached'),
    no_new_distinguishable_questions:t('已无可区分的新问题','No new distinguishable questions'),
    round_failed:t('本轮调查未完成','Investigation round incomplete')};
  if(stops[stop_reason])story(host,[[t('结束原因','Reason for stopping'),stops[stop_reason]]]);
  host.append(link(t('查看报告中的追加调查 ↗','View follow-ups in the report ↗'),data.followup.report));
}
function synthesisView(host){
  const {claim,evidence}=data.synthesis;
  host.append(el('h3','explanation-title',claim.title));
  p(host,claim.lede,'explanation-lede');
  for(const [key,title] of Object.entries(findingTitles)){
    const finding=claim.findings[key];
    if(!finding)continue;
    host.append(el('h3','section-title',title));
    p(host,finding.text);
    const list=el('ul','review-list');
    for(const id of finding.evidence_ids){
      const entry=evidence[id];
      const citation=link(entry.label,'reports/final.html#evidence-'+encodeURIComponent(id));
      const item=el('li');
      item.append(citation);
      list.append(item);
    }
    if(list.childElementCount)host.append(list);
  }
  host.append(button(t('打开完整报告 ↗','Open the full report ↗'),()=>openReport('final','claim'),'report-button'));
}
function programView(host,step){const records={scan:[['sampling.py',t('统计全语料各激活区间；均分500点配额，区间不足时将余额均分给其他区间。','Count each activation bin across the full corpus; allocate 500 positions equally and redistribute shortfalls.')],['evidence.py',t('合并极值、Exemplars、分位上下文和token统计；解释输入只使用product激活。','Combine extrema, exemplars, quantile contexts and token statistics; expose product activation to the explanation model.')]],initial:[['client.py',t('发送明确任务和JSON schema，保存结构化输出。','Send task instructions and a JSON schema, then save the structured output.')],['contracts.py',t('同一份条件说明用于报告正文和盲标规则。','Use the same condition descriptions for report prose and blind rules.')]],loops:[['labels.py',t('每批100点、5批并行；隐藏实测激活，读取完整前缀和邻近下文。','Five parallel batches of 100 points; hide measured activations and provide the full prefix and nearby continuation.')],['evidence.py',t('把标反点及超过对应语料极值50%却未覆盖的点，连同500点证据交给修订模型。','Provide sign reversals and uncovered points exceeding 50% of their corpus pole’s extreme, together with the 500 measured contexts.')],['interpretation.py',t('连续修订两次，每次使用紧前一轮的解释与盲标反馈。','Refine twice, each time using the immediately preceding explanation and prediction feedback.')]],freeze:[['interpretation.py',t('按准确率和覆盖率选择版本，同时保存它对应的盲标结果。','Select by accuracy and coverage, keeping the matching prediction set.')],['render.py',t('Spectrum和Inside the neuron使用同一批位置及同一套预测。','Spectrum and Inside the neuron use the same labeled positions and predictions.')]],experiments:[['run.py',t('组织3轮Search和独立Blind写作，向GPU提交测量请求。','Orchestrate three search rounds and independent blind writing, then submit GPU measurements.')],['controls.py',t('设计4组受控测试，测量后改进一次，并保留对应的实测结果。','Design four controlled tests, refine once after measurement and retain measured results.')],['measure.py',t('保留完整输入、明确的评分位置与有符号激活。','Preserve complete inputs, explicit scoring positions and signed activations.')]],report:[['paper_sections.py · interventions.py',t('测量40个归因案例、Family响应、3张图及匹配干预。','Measure 40 attribution cases, family responses, three graphs and matched interventions.')],['revision.py',t('分析实验依据和反例，独立评阅；保留冻结解释。','Analyze support and counterexamples and obtain an independent review, preserving the frozen explanation.')],['render.py',t('把结构化解释、实测数据、分析和评阅填入交互式HTML。','Assemble structured explanations, measurements, analysis and review in interactive HTML.')]],followup:[
  ['followup_investigation.py',t('从原始疑点出发规划，使用上一轮结果筛选可区分的新问题。','Plan from the original criticisms and use earlier results to select new distinguishable questions.')],
  ['followup_execute.py',t('读取已有证据或测量新输入，再记录独立判定和修订建议。','Review existing evidence or measure new inputs, then record a verdict and suggested revision.')]
],synthesis:[
  ['report_synthesis.py',t('综合冻结解释及各阶段的测量结果，为结论引用有稳定编号的证据。','Synthesize the frozen explanation and measured results, citing evidence by stable IDs.')],
  ['report_synthesis.js',t('依据证据编号跳转到最终报告中的相应观测。','Use evidence IDs to open the corresponding observations in the final report.')]
]};story(host,records[step]);}
async function overview(host,step){if(step==='scan'){story(host,[[t('输入','Input'),t('Qwen3-32B在Reasoning Mix 100M上的已保存激活和上下文。','Saved Qwen3-32B activations and contexts from Reasoning Mix 100M.')],[t('抽样','Sampling'),t('每个非空激活区间平均分配点数；只有全语料中不够配额时，才将余额均分给其他区间。','Allocate equally across nonempty activation bins. Redistribute only when a bin’s full-corpus population cannot fill its quota.')],[t('交给解释模型','Shown to the explanation model'),t('500个位置的实测product和上下文，加上两极极值、Exemplars、分位样本、token选择性、峰值统计及直接logit效应。','Measured product values and contexts for 500 positions, plus extrema, exemplars, quantile cases, token selectivity, peak statistics and direct logit effects.')]]);table(host,[t('激活区间','Activation bin'),t('全语料位置数','Corpus positions'),t('抽取点数','Sampled')],data.spectrum.corpus_hist.map((n,i)=>[data.spectrum.edges[i]+'…'+data.spectrum.edges[i+1],fmt(n),data.preparation.sample_bin_counts[i]]));}else if(step==='initial'){story(host,[[t('任务','Task'),t('找出反复出现的具体概念或表达模式，并用相反方向的样本检查条件是否过宽。','Identify recurring specific concepts or expression patterns, and check opposite-sign examples for conditions that are too broad.')],[t('上下文范围','Context'),t('结合完整前缀与紧邻的下文理解语义；空格、标点或普通词也可能承载已有上下文。','Use the full prefix and immediate continuation to recognize meaning. Spaces, punctuation and ordinary words may carry contextual information.')],[t('输出','Output'),t('正负极各最多5条条件。这些说明同时用于报告正文和盲标。','At most five conditions per pole, used for both report prose and blind labeling.')]]);explanation(host,data.rounds[0]);}else if(step==='loops'){roundCards(host);}else if(step==='freeze'){selectionTable(host);p(host,t('本次选择第1次修订：方向准确率为95.3%，覆盖率为34.5%。第2次修订的准确率更高，但覆盖率较低。','Refinement 1 is selected: 95.3% directional accuracy and 34.5% coverage. Refinement 2 is more accurate but covers fewer points.'),'selection-reason');explanation(host,selected());p(host,t('从这里开始，Search、Blind和后续报告分析都使用这份解释。','From this point, search, blind writing and report analysis all use this explanation.'));}else if(step==='experiments'){experimentView(host);}else if(step==='followup'){followupView(host);}else if(step==='synthesis'){synthesisView(host);}else{story(host,[[t('归因与干预','Attribution and interventions'),t('对自然语料与构造文本的40个案例进行归因，测量Family响应，并对3个图根做匹配干预。','Attribute 40 natural and authored cases, measure family responses and perform matched interventions on three graph roots.')],[t('报告分析','Report analysis'),t('模型根据实验记录分析支持、反例和未解释的模式；总体解释及其桶规则保持冻结。','The model analyzes experimental support, counterexamples and unresolved patterns while retaining the frozen explanation and conditions.')],[t('最终产物','Final output'),t('固定程序组装实测图表、分析、独立评阅和成本记录。','Code assembles measured plots, analysis, independent review and recorded costs.')]]);const review=data.review;host.append(el('h3','',t('独立评阅','Independent review')));p(host,review.one_line);if(review.concerning?.length){const list=el('ul','review-list');review.concerning.forEach(x=>list.append(el('li','',x)));host.append(list);}reportExtras(host,false);host.append(button(t('打开最终交互报告 ↗','Open the final interactive report ↗'),()=>openReport('final'),'report-button'));}}
async function outputs(host,step){if(step==='scan'){code(host,t('实际抽样结果','Recorded sampling result'),data.preparation);details(host,t('查看极值与分位样本的证据节选','Inspect extreme and quantile evidence excerpts'),d=>code(d,t('证据节选','Evidence excerpt'),data.evidence));}else if(step==='initial')explanation(host,data.rounds[0]);else if(step==='loops')roundCards(host);else if(step==='freeze'){selectionTable(host);code(host,t('最终冻结的解释','Frozen explanation'),selected().explanation);}else if(step==='experiments')experimentView(host);else if(step==='followup')code(host,t('后续验证记录','Follow-up investigation record'),data.followup);else if(step==='synthesis')code(host,t('综合结论及其证据','Synthesis and cited evidence'),data.synthesis);else{reportExtras(host,true);details(host,t('报告分析输出','Report-analysis output'),d=>code(d,t('报告分析','Report analysis'),data.analysis));details(host,t('独立评阅输出','Independent review output'),d=>code(d,t('独立评阅','Independent review'),data.review));details(host,t('本次运行的成本记录','Recorded investigation costs'),d=>code(d,'Interpretation Costs',data.costs));host.append(button(t('打开最终报告 ↗','Open final report ↗'),()=>openReport('final'),'report-button'));}}
function preview(step){const host=$('#preview');host.replaceChildren();if(step.wide)return;const card=el('div','preview-card');card.append(el('div','preview-title',step.id==='initial'?t('初步解释的桶标签','Initial response conditions'):t('这次运行的HTML','Recorded HTML')));if(step.id==='initial'){const box=el('div','preview-rules');for(const rule of data.rounds[0].rules.filter(r=>r.predicted_pole!=='uncertain')){const line=el('p');line.append(el('i','swatch '+rule.predicted_pole),document.createTextNode(rule.name));box.append(line);}card.append(box);}else{const kind=step.id==='scan'?'evidence':'final';const frame=el('iframe','live-preview');frame.title='Recorded neuron report';frame.src='reports/'+kind+'.html';frame.loading='lazy';card.append(frame);const actions=el('div','preview-actions');actions.append(button(t('展开阅读 ↗','Open report ↗'),()=>openReport(kind)));card.append(actions);}host.append(card);}
async function renderTab(){const token=++revision,step=steps[current],host=$('.tabbody');host.replaceChildren();try{if(activeTab==='overview')await overview(host,step.id);else if(activeTab==='prompt')await promptView(host,step.id,token);else if(activeTab==='actions')programView(host,step.id);else await outputs(host,step.id);}catch(error){if(token===revision)p(host,error.message,'error');}}
function select(index,tab='overview'){current=Math.max(0,Math.min(steps.length-1,index));activeTab=tab;const step=steps[current];$('.workspace').classList.toggle('wide-step',!!step.wide);$('#step-counter').textContent=(current+1)+'／'+steps.length;if(!document.body.classList.contains('cover-open'))history.replaceState(null,'','#'+step.id);document.querySelectorAll('.step-button').forEach((b,i)=>{b.classList.toggle('active',i===current);b.classList.toggle('past',i<current);});const detail=$('#detail');detail.replaceChildren();const heading=el('div','step-heading'),eyebrow=el('div','step-eyebrow',t('第','Step ')+(current+1)+(language==='zh'?'步':''));eyebrow.append(el('span','role',step.role));heading.append(eyebrow,el('h2','',step.title),el('p','',step.summary));detail.append(heading);const nav=el('div','tabs');Object.entries(tabs).forEach(([key,label])=>{const b=button(label,()=>{activeTab=key;nav.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));renderTab();},key===activeTab?'active':'');nav.append(b);});detail.append(nav,el('div','tabbody'));const controls=el('div','step-controls');const back=button(t('← 上一步','← Previous'),()=>select(current-1)),next=button(t('下一步 →','Next →'),()=>select(current+1),'primary');back.disabled=current===0;next.disabled=current===steps.length-1;controls.append(back,el('span','middle',(current+1)+'／'+steps.length),next);detail.append(controls);preview(step);renderTab();$('#playback-caption').textContent=(current+1)+'／'+steps.length+' · '+step.nav;const progress=$('#playback-progress');progress.replaceChildren();steps.forEach((s,i)=>progress.append(button(s.nav,()=>select(i),i===current?'active':i<current?'past':'')));$('#demo-prev').disabled=current===0;$('#demo-next').textContent=current===steps.length-1?t('从头再看','Start over'):t('下一步：','Next: ')+steps[current+1].nav;updateLanguageLinks();if(!document.body.classList.contains('cover-open'))window.scrollTo({top:0});}
function start(index=0){document.body.classList.remove('cover-open');select(index);}
function showCover(){document.body.classList.add('cover-open');history.replaceState(null,'',location.pathname+location.search);window.scrollTo({top:0});}
function init(){const nav=$('#step-nav');steps.forEach((s,i)=>{const b=button('',()=>start(i),'step-button');b.innerHTML='<span class="num">'+String(i+1).padStart(2,'0')+'</span><span>'+esc(s.nav)+'</span>';nav.append(b);});$('#start-demo').onclick=()=>start();$('#back-cover').onclick=showCover;$('#home-link').onclick=e=>{e.preventDefault();showCover();};$('#demo-prev').onclick=()=>select(current-1);$('#demo-next').onclick=()=>select(current===steps.length-1?0:current+1);$('#close-reading').onclick=()=>$('#reading-dialog').close();$('#reading-dialog').addEventListener('close',()=>{$('#reading-content').replaceChildren();});$('#reading-dialog').addEventListener('click',e=>{if(e.target!==e.currentTarget)return;const r=e.currentTarget.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.currentTarget.close();});$('#cover-zoom').onclick=()=>openReport(coverSnapshot);$('#about-demo').onclick=()=>{const body=modal(t('关于这段记录','About this recording'));p(body,t('这里回放Colorado神经元在Reasoning Mix 100M上的一次已完成研究。文字、桶标签、模型输出及数值来自保存的运行记录，图表由这些实测值绘制。','This walkthrough follows a completed Colorado neuron investigation on Reasoning Mix 100M. Explanations, conditions, model outputs and numerical values come from saved records; plots are drawn from those measurements.'));p(body,t('各轮使用同一批500个位置。因为它们用于修订解释，准确率和覆盖率反映这批样本上的表现。','All rounds use the same pool of 500 positions. Because these positions inform refinement, accuracy and coverage describe performance on that sample pool.'));};document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>start(steps.findIndex(s=>s.id===b.dataset.jump)));document.querySelectorAll('[data-cover-snapshot]').forEach(b=>b.onclick=()=>{coverSnapshot=b.dataset.coverSnapshot;$('#cover-paper').src='reports/'+coverSnapshot+'.html';document.querySelectorAll('[data-cover-snapshot]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',String(x===b));});});document.addEventListener('keydown',e=>{if($('#reading-dialog').open||['INPUT','SELECT','TEXTAREA'].includes(e.target.tagName))return;if(e.key==='ArrowRight'){e.preventDefault();start(Math.min(current+1,steps.length-1));}else if(e.key==='ArrowLeft'){e.preventDefault();start(Math.max(current-1,0));}});const index=steps.findIndex(s=>s.id===location.hash.slice(1));if(index>=0)start(index);else select(0);}
get('content/episode.json').then(episode=>{data=episode;
  if(data.followup)steps.push({
    id:'followup',wide:true,role:t('LLM＋固定程序','LLM + Python'),
    nav:t('后续验证','Follow-up investigation'),
    title:t('逐轮验证报告中的疑点','Investigate the report’s criticisms'),
    summary:t('查看每轮提出的问题、实测观测、判定和修订建议。','Review questions, measured observations, verdicts and suggested revisions by round.')
  });
  if(data.synthesis)steps.push({
    id:'synthesis',wide:true,role:t('LLM＋固定程序','LLM + Python'),
    nav:t('证据综合','Evidence synthesis'),
    title:t('查看结论及其依据','Read the claim and its evidence'),
    summary:t('综合报告串联各阶段结果，可从每条结论跳转到引用的实测证据。','The synthesis brings the results together and links each finding to its cited evidence.')
  });
  init();}).catch(error=>{$('#detail').textContent=error.message;});
})();
