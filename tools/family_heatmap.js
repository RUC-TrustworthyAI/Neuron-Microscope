  if (fam.length && Ms.length) {
    let sel = 0;
    const holder = $('div');
    const select = $('select', {class:'sel'}, Ms.map((m,i)=>
      $('option',{value:i},(m.prompt || `sentence ${i+1}`).slice(0,96))));
    const drawHeat = () => {
      holder.replaceChildren();
      const m=Ms[sel],T=m.tokens || [];
      const L=300,cw=Math.max(12,Math.min(34,Math.floor(780/Math.max(1,T.length))));
      const rh=22,top=6,botLab=70,W=L+cw*T.length+8,H=top+rh*fam.length+botLab;
      const svg=svgEl('svg',{viewBox:`0 0 ${W} ${H}`,width:'100%',
        style:`display:block;max-width:${W}px`});
      fam.forEach((u,i)=>{
        const vals=(m.readouts || []).find(r=>r.layer===u.layer && r.neuron===u.neuron)?.values || [];
        const mx=vals.reduce((value,x)=>Math.max(value,Math.abs(x)),1e-9),y=top+i*rh;
        const label=svgEl('text',{x:L-8,y:y+rh/2+4,'text-anchor':'end','font-size':12,
          'font-weight':isSelf(u)?700:400,fill:'#1d1d1f'},
          `${unitId(u)} · ${(isSelf(u)?'this neuron':(u.label || '')).slice(0,30)}`);
        label.append(svgEl('title',null,`${unitId(u)} · ${u.label || ''}`));svg.append(label);
        T.forEach((t,j)=>{
          const v=vals[j] || 0;
          const cell=svgEl('rect',{x:L+j*cw,y,width:cw-1.5,height:rh-1.5,rx:2,
            fill:hexA(v>=0?POS:NEG,0.04+0.9*Math.abs(v)/mx)});
          cell.append(svgEl('title',null,`${unitId(u)} · ${JSON.stringify(t)} · ${fmt(v)}`));
          svg.append(cell);
        });
      });
      T.forEach((t,j)=>{
        const x=L+j*cw+cw/2+3,y=top+rh*fam.length+8;
        svg.append(svgEl('text',{x,y,'font-size':10,fill:'#4a4a4f',
          transform:`rotate(-60 ${x} ${y})`,'text-anchor':'end'},t.trim() || '␣'));
      });
      holder.append(svg);
    };
    select.onchange=()=>{sel=+select.value;drawHeat();};drawHeat();
    figs.push(fig($('div',null,Ms.length>1?$('div',{class:'ctlrow'},'Sentence ',select):null,holder),
      'Each row is one unit of the family, normalised to its own largest |activation| on the sentence, so the token pattern is comparable even when scales differ. This neuron’s row is bold.'));
  }
