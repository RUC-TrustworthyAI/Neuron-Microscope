const headlineFigure = () => {
  const Hd=B.headline;if(!has(Hd))return null;
  const wrap=$('div',{class:'headline-poles'}),row=$('div',{class:'headline'});
  const panels=has(Hd.poles)?Hd.poles:[{ladder:Hd.ladder,strip:Hd.strip}];
  const L=Hd.comparison || panels.find(p=>p.ladder)?.ladder;
  if(has(L)&&has(L.fillers)){
    const F=[...L.fillers].sort((a,b)=>(b.activation||0)-(a.activation||0));
    const lo=Math.min(0,...F.map(f=>f.activation||0)),hi=Math.max(0,...F.map(f=>f.activation||0));
    const lines=text=>String(text||'').split(/\s+/).reduce((out,word)=>{
      if(!out.length||out[out.length-1].length+word.length+1>28)out.push(word);
      else out[out.length-1]+=' '+word;return out;},[]);
    const labels=F.map(f=>lines(f.tag)),LB=210,W=520,RB=56;
    const rh=24+13*(Math.max(1,...labels.map(x=>x.length))-1),H=rh*F.length+8;
    const X=v=>LB+(v-lo)/((hi-lo)||1)*(W-LB-RB);
    const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',class:'ladder',style:'display:block'});
    const hover=$('div',{class:'hover'}),show=f=>{
      const ta=has(f.tokens)?zipToks(f.tokens,f.values):null;
      const pk=ta?(Number.isInteger(f.peak_index)?f.peak_index:argmaxAbs(f.values)):0;
      hover.replaceChildren($('span',{class:'k'},f.tag),ta?toks(ta,pk,null,(f.activation||0)<0):$('span',{class:'toks'},f.full_text||f.text||''));
    };
    svg.append(svgEl('line',{x1:X(0),x2:X(0),y1:2,y2:H-4,stroke:'#c9c9ce'}));
    F.forEach((f,i)=>{
      const y=4+i*rh,v=f.activation||0,g=svgEl('g',{class:'row'});
      g.append(svgEl('rect',{class:'bg',x:0,y,width:W,height:rh,fill:'transparent'}));
      labels[i].forEach((line,j)=>g.append(svgEl('text',{x:LB-8,y:y+rh/2+4+(j-(labels[i].length-1)/2)*13,'text-anchor':'end','font-size':12,fill:'#1d1d1f'},line)));
      g.append(svgEl('rect',{x:Math.min(X(0),X(v)),y:y+rh/2-7,width:Math.max(1,Math.abs(X(v)-X(0))),height:14,rx:2,fill:v>=0?POS:NEG}));
      g.append(svgEl('text',{x:v>=0?X(v)+6:X(0)+6,y:y+rh/2+4,'font-size':12,fill:'#4a4a4f'},fmt(v)));
      g.onmouseenter=()=>show(f);svg.append(g);
    });
    if(F.length)show(F[0]);
    let template=null;
    if(L.template){template=$('div',{class:'tmpl'});L.template.split(/(\{[A-Za-z_]+\})/).forEach(t=>template.append(/^\{[A-Za-z_]+\}$/.test(t)?$('span',{class:'slot'},t.slice(1,-1)):t));}
    row.append($('div',{class:'panel'},fig($('div',null,svg,template,hover),L.caption||null)));
  }
  const S=panels.find(p=>p.strip)?.strip || Hd.strip;
  if(has(S)&&has(S.tokens)){
    const values=S.values||[],mx=Math.max(...values.map(Math.abs),1e-9);
    const peak=Number.isInteger(S.anchor_index)?S.anchor_index:argmaxAbs(values);
    const strip=$('div',{class:'strip'});
    S.tokens.forEach((token,i)=>{
      const v=values[i]||0,h=40*Math.abs(v)/mx;
      const bar=$('i',{style:v>=0?`bottom:16px;height:${h}px;background:${POS}`:`top:44px;height:${Math.min(h,14)}px;background:${NEG};border-radius:0 0 2px 2px`,title:fmt(v)});
      strip.append($('span',{class:'stk'},$('span',{class:'bars'},$('i',{class:'base'}),Math.abs(v)/mx>.02?bar:null),
        $('span',{class:'tk'+(i===peak?' peak'+(v<0?' n':''):''),title:fmt(v)},token)));
    });
    row.append($('div',{class:'panel'},fig(strip,S.caption||null)));
  }
  if(row.children.length===1)row.classList.add('single');if(row.children.length)wrap.append(row);
  return wrap.children.length ? wrap : null;
};
