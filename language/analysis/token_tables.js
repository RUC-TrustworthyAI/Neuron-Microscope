
(() => {
  const views=JSON.parse(document.getElementById('token-selectivity-data').textContent);
  const meta=JSON.parse(document.getElementById('token-selectivity-summary').textContent);
  const make=(tag,cls,text)=>{const n=document.createElement(tag);if(cls)n.className=cls;if(text!==undefined)n.textContent=text;return n};
  const section=make('section','sec');section.id='token-selectivity';
  const text=make('div','text');text.append(make('div','lab','Token selectivity'));
  const why=make('div','why');why.append(make('p',null,"Which tokens consistently reach a fraction of this neuron's positive or negative extreme?"));text.append(why);section.append(text);
  const wide=make('div','figs wide'),grid=make('div','ts-grid');wide.append(grid);section.append(wide);
  const fmt=n=>n.toLocaleString('en-US');
  for(const pole of ['positive','negative']){
    const choices=views.filter(v=>v.view.mode===pole),panel=make('div','ts-panel ts-'+pole),heading=make('div','ts-heading'),top=make('div','ts-topline');
    top.append(make('div','ts-pole',pole==='positive'?'Positive tail':'Negative tail'));const threshold=make('div','ts-threshold');top.append(threshold);heading.append(top);
    const controls=make('div','ts-controls'),seg=make('div','seg');seg.setAttribute('role','group');seg.setAttribute('aria-label',pole+' activation threshold');controls.append(seg,make('span','quiet','Top 80 tokens'));heading.append(controls);panel.append(heading);
    const scroll=make('div','ts-scroll');scroll.tabIndex=0;scroll.setAttribute('role','region');scroll.setAttribute('aria-label',pole+' token rankings');
    const table=make('table'),thead=make('thead'),tr=make('tr');
    const columns=[['#','ts-rank'],['Token','ts-token'],['Count','ts-count ts-num'],['Mean','ts-mean ts-num'],['Hits','ts-hits ts-num'],['Fraction','ts-share ts-num']];
    for(const [label,cls] of columns){const th=make('th',cls,label);th.scope='col';tr.append(th)}thead.append(tr);table.append(thead);const body=make('tbody');table.append(body);scroll.append(table);panel.append(scroll);grid.append(panel);
    const draw=(selected)=>{
      const chosen=choices[selected],v=chosen.view;threshold.textContent='Product '+(pole==='positive'?'> ':'< ')+v.threshold;
      seg.querySelectorAll('button').forEach((b,i)=>{b.classList.toggle('on',i===selected);b.setAttribute('aria-pressed',String(i===selected))});body.replaceChildren();
      chosen.rows.forEach((r,i)=>{const row=make('tr');row.dataset.tokenId=r[0];row.dataset.count=r[2];row.dataset.hits=r[v.column];const token=JSON.stringify(r[1]);
        row.append(make('td','ts-rank',i+1));const t=make('td','ts-token');t.title=token;t.append(make('code',null,token));row.append(t,make('td','ts-count ts-num',fmt(r[2])),make('td','ts-mean ts-num',r[3].toFixed(2)),make('td','ts-hits ts-num',fmt(r[v.column])));
        const cell=make('td','ts-share ts-num'),rate=make('div','ts-rate'),fraction=r[v.column]/r[2],bar=make('i');bar.style.width=(100*fraction)+'%';rate.title=fmt(r[v.column])+' / '+fmt(r[2]);rate.append(bar,make('span',null,(100*fraction).toFixed(2)+'%'));cell.append(rate);row.append(cell);body.append(row);
      });scroll.scrollTop=0;
    };
    choices.forEach((choice,i)=>{const b=make('button',null,(choice.view.fraction*100)+'%');b.type='button';b.setAttribute('aria-label',pole+' '+(choice.view.fraction*100)+'% of extreme');b.onclick=()=>draw(i);seg.append(b)});
    draw(1);
  }
  wide.append(make('p','ts-method',fmt(meta.tokens)+' tokens · tokens with at least '+meta.minimum_count+' occurrences · positive maximum '+meta.positive_max+', negative minimum '+meta.negative_min+'. Fraction = hits / all occurrences of that token.'));
  const distribution=document.getElementById('distribution');if(!distribution)throw Error('Activation distribution section missing');distribution.after(section);
  const nav=document.getElementById('nav'),after=nav.querySelector('a[href="#distribution"]'),link=make('a',null,'Tokens');link.href='#token-selectivity';if(after)after.after(link);else nav.append(link);
  if(location.hash==='#token-selectivity')requestAnimationFrame(()=>section.scrollIntoView({block:'start'}));
})();
