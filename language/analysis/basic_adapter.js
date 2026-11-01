/* Display the shared activation fields; saved explanation records are not rendered. */
(() => {
  const missing='—', tokenCount=Number(B.distribution.n_tokens).toLocaleString();
  const order=['claim','custom','graphs','distribution','spectrum','exemplars','search','clusters','anatomy','ablation','family','circuit','tests'];
  const absent={
    custom:['Figures for this neuron',['Measured figures']],
    graphs:['Three graphs',['Attribution graphs']],
    spectrum:['Activation spectrum',['Semantic groups','Labeled spectrum']],
    search:['Activation search',['Search regions','Scored probes','Best positive','Best negative']],
    clusters:['Clusters',['Clusters','Routes','Route factorization']],
    ablation:['What it does',['Direct logit effect','Neuron ablation','Random controls']],
    family:['Where it sits',['Neuron family','Joint measurements']],
    circuit:['Circuit',['Upstream','Downstream','Logits']],
    tests:['Tests',['Blind maximum','Blind minimum','Route prediction']]
  };
  Object.entries(absent).forEach(([id,[label,fields]])=>{
    if(!document.getElementById(id))sec({id,wide:true},label,null,
      $('div',{class:'tiles'},fields.map(name=>kTile(name,null,{text:missing}))));
  });
  // These are the same peak-census fields; unavailable measurements remain —.
  const claim=document.getElementById('claim');
  if(!claim.querySelector('.census')){
    const field=name=>$('div',null,$('div',{class:'sub'},name),$('span',{class:'tile'},missing));
    claim.querySelector(':scope > .figs').append(fig($('div',{class:'census'},
      field('Token kind at the peak'),field('Most common peak tokens'))));
  }
  const main=document.getElementById('main'),nav=document.getElementById('nav');
  order.forEach(id=>{
    const node=document.getElementById(id);if(node)main.append(node);
    const link=nav.querySelector('a[href="#'+id+'"]');if(link)nav.append(link);
  });
  const hist=document.getElementById('distribution');
  const why=hist.querySelector('.why');if(why)why.replaceChildren(P('Saved activation distribution across the natural corpus.'));
  hist.querySelectorAll('.cap').forEach(node=>{
    if(node.textContent.includes('four GPUs'))node.textContent='Counts over all '+tokenCount+' saved tokens, grouped by native BF16 value.';
    if(node.textContent.includes('only the peak token has a value'))node.textContent=missing;
  });
  hist.querySelectorAll('svg text').forEach(node=>{
    if(node.textContent==='bold ticks = exemplar bands shown below')node.textContent='quantile bands shown below';
  });
  if(B.bigscan?.hist_centers?.length===1){
    const chart=hist.querySelectorAll('svg')[1];
    if(chart)chart.append(
      svgEl('line',{'data-constant-frequency':'true',x1:514,x2:514,y1:182,y2:16,stroke:'#4a4a4f','stroke-width':2}),
      svgEl('circle',{cx:514,cy:16,r:4,fill:'#4a4a4f'}),
      svgEl('text',{x:524,y:30,'font-size':12},B.bigscan.hist[0].toLocaleString()+' tokens at '+B.bigscan.hist_centers[0]));
  }
  const anatomy=document.getElementById('anatomy');
  const awhy=anatomy.querySelector('.why');if(awhy)awhy.replaceChildren(P('Activation = SiLU(raw gate) × value.'));
  anatomy.querySelectorAll('.cap').forEach(node=>{
    if(node.textContent.includes('one per labeled token'))node.textContent='Raw gate and value at '+B.anatomy.n.toLocaleString()+' saved token positions.';
  });
  anatomy.querySelectorAll('.sub').forEach(node=>{
    if(node.textContent!=='One sentence, three signals')return;
    const figure=node.closest('figure');
    if(figure)figure.replaceChildren($('div',null,node,$('span',{class:'tile'},missing)));
  });
  const exemplars=document.getElementById('exemplars');
  const ewhy=exemplars.querySelector('.why');
  if(ewhy)ewhy.replaceChildren(P('Natural-corpus maxima, minima, and random samples by quantile band.'));
  exemplars.querySelectorAll('.pane').forEach(node=>{
    if(!node.querySelector('.ex'))node.append($('span',{class:'tile'},missing));
  });
  document.querySelectorAll('.tile').forEach(node=>{
    if(node.textContent==='–')node.textContent=missing;
    else if(node.textContent.endsWith(' vs –'))node.textContent=node.textContent.slice(0,-1)+missing;
  });
  const contexts=JSON.parse(document.getElementById('local-contexts').textContent);
  const figures=Array.from(exemplars.querySelector(':scope > .figs').children);
  const groups=[{root:figures[0],items:B.exemplars.top},{root:figures[1],items:B.exemplars.bottom},
    ...Array.from(exemplars.querySelectorAll('.pane')).map((root,i)=>({root,items:Object.values(B.exemplars.bands)[i]}))];
  let duplicate=0;
  function decorate(){
    groups.forEach(group=>group.root.querySelectorAll('.ex').forEach((row,i)=>{
      if(row.dataset.contextReady)return;
      const item=group.items[i];if(!item||!contexts[item.id])return;
      row.dataset.contextReady='true';row.dataset.pointId=item.id;
      row.querySelectorAll('.t.peak').forEach(t=>t.classList.toggle('n',item.activation<0));
      const id='case-'+item.id;row.id=document.getElementById(id)?id+'-copy-'+(++duplicate):id;
    }));
  }
  decorate();document.addEventListener('click',()=>setTimeout(decorate,0));
  const target=location.hash.startsWith('#case-')?document.querySelector(location.hash):null;
  if(target)target.scrollIntoView();
})();
