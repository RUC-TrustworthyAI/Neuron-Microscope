function combinedPeakCensus(positive, negative) {
  const sources=[positive,negative].filter(x=>x && x.n);
  if (!sources.length) return null;
  const combine=(field,key)=>{
    const counts=new Map();
    sources.forEach(source=>(source[field] || []).forEach(row=>
      counts.set(row[key],(counts.get(row[key]) || 0)+(row.n || 0))));
    return [...counts].map(([label,n])=>({[key]:label,n}));
  };
  return {n:sources.reduce((n,x)=>n+x.n,0),kinds:combine('kinds','kind'),tokens:combine('tokens','token'),
    caption:[positive?.n ? `${positive.n} positive` : '',negative?.n ? `${negative.n} negative` : ''].filter(Boolean).join(' and ')+
      ' peak positions, pooled by token kind.'};
}
function peakCensusFigure(PC, label) {
  if (!has(PC) || (!has(PC.kinds) && !has(PC.tokens))) return null;
  const kinds=[...(PC.kinds || [])].sort((a,b)=>(b.n||0)-(a.n||0));
  const km=Math.max(...kinds.map(k=>k.n||0),1);
  const bars=$('div',{class:'kbars'},kinds.slice(0,7).map(k=>$('div',{class:'kb'},
    $('span',{class:'kl'},k.kind),$('span',{class:'kt'},$('i',{style:`width:${100*(k.n||0)/km}%`})),
    $('span',{class:'kn'},k.n))));
  const tokens=[...(PC.tokens || [])].sort((a,b)=>(b.n||0)-(a.n||0)).slice(0,14);
  const chips=$('div',{class:'chips',style:'margin:0'},tokens.map(t=>$('span',{class:'tokchip'},
    JSON.stringify(t.token),' ',$('b',{style:'color:var(--ink);font-weight:500'},t.n))));
  const node=fig($('div',null,label?$('div',{class:'sub'},label):null,$('div',{class:'census'},
    $('div',null,$('div',{class:'sub'},'Token kind at the peak'),bars),
    $('div',null,$('div',{class:'sub'},'Most common peak tokens'),chips))),
    PC.caption || `${PC.n} token positions · ${PC.distinct_documents} distinct documents.`);
  if (PC.note) node.title=PC.note;
  return node;
}
