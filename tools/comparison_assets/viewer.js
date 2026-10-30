(() => {
  'use strict';
  if (window.self !== window.top || document.getElementById('np-compare-toggle')) return;
  const script = document.currentScript;
  const neuron = script.dataset.neuron;
  const reference = new URL(script.dataset.reference, location.href).href;
  const original = script.dataset.original;
  const make = (tag, attributes = {}, text) => {
    const element = document.createElement(tag);
    for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, value);
    if (text !== undefined) element.textContent = text;
    return element;
  };
  const panelIcon = '<svg viewBox="0 0 20 20" fill="none" stroke="currentColor" stroke-width="1.4" aria-hidden="true"><rect x="2.2" y="3.5" width="15.6" height="13" rx="2"/><path d="M9 3.5v13"/><path class="np-compare-arrow" d="M4.5 8l2 2-2 2"/></svg>';
  const externalIcon = '<svg viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.3" aria-hidden="true"><path d="M9 2h5v5M14 2 7 9M7 3H3a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1V9"/></svg>';
  const toggle = make('button', {id:'np-compare-toggle', type:'button', 'aria-label':'Compare reports', 'aria-expanded':'false', 'aria-controls':'np-compare-shell', title:'Compare reports'});
  toggle.innerHTML = panelIcon;
  const toggleLabel = make('span', {}, 'Compare reports');
  const toggleArrow = toggle.querySelector('.np-compare-arrow');
  toggle.append(toggleLabel);
  const setToggleState = expanded => {
    const label = expanded ? 'Close comparison' : 'Compare reports';
    toggleLabel.textContent = label;
    toggleArrow.setAttribute('d', expanded ? 'M6.5 8l-2 2 2 2' : 'M4.5 8l2 2-2 2');
    toggle.setAttribute('aria-expanded', String(expanded));
    toggle.setAttribute('aria-label', label);
    toggle.title = label;
  };
  document.body.append(toggle);
  const shell = make('div', {id:'np-compare-shell', role:'dialog', 'aria-label':'Compare the same neuron', 'aria-modal':'true', hidden:''});
  const leftHeading = make('div', {id:'np-compare-left-heading', class:'np-compare-heading'});
  leftHeading.append(make('strong', {}, 'Published report'), make('small', {}, neuron));
  const link = make('a', {href:original, target:'_blank', rel:'noopener noreferrer', title:'Open published report', 'aria-label':'Open published report'});
  link.innerHTML = externalIcon; leftHeading.append(link);
  const rightHeading = make('div', {id:'np-compare-right-heading', class:'np-compare-heading'});
  rightHeading.append(make('strong', {}, 'Replication report'), make('small', {}, neuron));
  const close = make('button', {id:'np-compare-close', type:'button', title:'Close comparison', 'aria-label':'Close comparison'}, '\u00d7');
  rightHeading.append(close);
  const divider = make('div', {id:'np-compare-divider', role:'separator', tabindex:'0', 'aria-label':'Resize comparison panes', 'aria-orientation':'vertical', 'aria-valuemin':'25', 'aria-valuemax':'75', 'aria-valuenow':'50'});
  const panes = ['left','right'].map(side => {
    const pane = make('div', {id:'np-compare-'+side+'-pane', class:'np-compare-pane'});
    pane.append(make('div', {class:'np-compare-loading'}, 'Loading report\u2026'));
    return pane;
  });
  shell.append(leftHeading, rightHeading, divider, ...panes);
  document.body.append(shell);
  const frames = []; let opened = false, ready = false, pairs = [], alignTimer = 0, alignFrame = 0;
  let savedPosition = null, pendingPosition = null;
  const expected = [null, null], syncFrames = [0, 0];
  const anchorKey = element => element.dataset.npComparisonKey || element.id;
  const getAnchors = doc => {
    const anchors = [];
    const add = (element, key) => {
      if (!element || anchors.includes(element)) return;
      if (key) element.dataset.npComparisonKey=key;
      anchors.push(element);
    };
    const head=doc.getElementById('head');
    add(head);
    if (head) {
      add(head.querySelector('.lede'),'head-lede');
      add(head.querySelector('.pr-box'),'head-agent-read');
      add(head.querySelector(':scope > .headline-poles, :scope > .headline'),'head-headline');
      add(head.querySelector('.metarow'),'head-meta');
    }
    doc.querySelectorAll('main > section[id], main > .prf[id]').forEach(section=>{
      add(section);
      add(section.querySelector(':scope > .figs'),section.id+'-figures');
    });
    const claim=doc.getElementById('claim');
    if (claim) claim.querySelectorAll('.census').forEach((census,index)=>{
      add(census,'claim-peak-census-'+index);
      census.querySelectorAll(':scope > div > .sub').forEach((title,column)=>{
        add(title,'claim-peak-census-'+index+'-title-'+column);
      });
    });
    const exemplars=doc.getElementById('exemplars');
    if (exemplars) {
      const titles=[...exemplars.querySelectorAll('.sub')];
      const roles=[
        ['exemplars-top',/^Top activations\b/i],
        ['exemplars-bottom',/^Bottom activations\b/i],
        ['exemplars-bands',/^Random samples by quantile band\b/i]
      ];
      roles.forEach(([key,pattern])=>{
        add(titles.find(title=>pattern.test(title.textContent.trim())),key);
      });
    }
    const clusters=doc.getElementById('clusters');
    if (clusters) {
      const raw=[...clusters.querySelectorAll('details > summary')].find(
        title=>/^Raw factorization\b/i.test(title.textContent.trim()));
      add(raw?.parentElement,'clusters-raw-factorization');
    }
    const family=doc.getElementById('family');
    if (family) {
      const heatmap=family.querySelector('svg:not(.wires)');
      add(heatmap?.parentElement,'family-heatmap');
      add(family.querySelector('.neighwrap'),'family-neighbourhood');
    }
    const routeTitle=[...doc.querySelectorAll('#tests .sub')].find(
      node=>/^Route prediction$/i.test(node.textContent.trim()));
    add(routeTitle?.closest('figure') || routeTitle,'tests-route-prediction');
    const criticism=doc.getElementById('criticism');
    if (criticism) {
      let end=doc.querySelector('[data-np-criticism-end]');
      if (!end) {
        end=doc.createElement('div');end.dataset.npCriticismEnd='';
        end.setAttribute('aria-hidden','true');criticism.after(end);
      }
      add(end,'criticism-end');
    }
    return anchors.sort((a,b)=>a.compareDocumentPosition(b)&4?-1:1);
  };
  const topOf = element => element.getBoundingClientRect().top + element.ownerDocument.defaultView.scrollY;
  const capture = win => {
    const doc = win.document, anchors = getAnchors(doc);
    const inset = (doc.getElementById('nav')?.getBoundingClientRect().height || 0) + 20;
    const y = win.scrollY + inset;
    if (anchors[0] && win.scrollY < topOf(anchors[0])) return {id:'__top', y:win.scrollY};
    let anchor = anchors[0];
    for (const current of anchors) if (topOf(current) <= y) anchor = current;
    if (!anchor) return null;
    if (win.scrollY < topOf(anchor)) return {id:anchorKey(anchor), heading:true};
    const next = anchors[anchors.indexOf(anchor)+1];
    const extent = next ? topOf(next)-topOf(anchor) : Math.max(anchor.offsetHeight,1);
    return {id:anchorKey(anchor), fraction:Math.max(0,Math.min(1,(win.scrollY-topOf(anchor))/Math.max(extent,1)))};
  };
  const positionAt = (win, position) => {
    if (!position) return 0;
    if (position.id==='__top') return position.y || 0;
    const anchors = getAnchors(win.document), anchor = anchors.find(element=>anchorKey(element)===position.id);
    if (!anchor) return 0;
    if (position.heading) return Math.max(0,topOf(anchor)-(win.document.getElementById('nav')?.getBoundingClientRect().height || 0)-16);
    const next = anchors[anchors.indexOf(anchor)+1];
    const extent = next ? topOf(next)-topOf(anchor) : Math.max(anchor.offsetHeight,1);
    return Math.max(0,topOf(anchor)+position.fraction*Math.max(extent,1));
  };
  const sync = (index) => {
    if (!opened || !ready) return;
    const win = frames[index].contentWindow, y = win.scrollY;
    if (expected[index] !== null && Math.abs(y-expected[index]) < 2) { expected[index] = null; return; }
    expected[index] = null;
    // A new user scroll takes precedence over a queued update from the other pane.
    cancelAnimationFrame(syncFrames[1-index]);
    cancelAnimationFrame(syncFrames[index]);
    syncFrames[index] = requestAnimationFrame(() => {
      const other = 1-index;
      expected[other] = win.scrollY;
      frames[other].contentWindow.scrollTo({top:win.scrollY, behavior:'instant'});
    });
  };
  function align() {
    if (!opened || !ready) return;
    const inset = Math.max(...frames.map(frame => frame.contentDocument.getElementById('nav')?.getBoundingClientRect().height || 0))+16;
    frames.forEach(frame => frame.contentDocument.documentElement.style.setProperty('--np-comparison-inset',inset+'px'));
    const right = frames[1].contentWindow, position = pendingPosition || capture(right);
    const restorePosition = !!pendingPosition;pendingPosition=null;
    let changed = false;
    // Measure each anchor after prior changes: nested and side-by-side blocks need not share a linear flow.
    pairs.forEach(pair => {
      const natural=pair.elements.map((element,side)=>
        topOf(element)-(parseFloat(pair.spacers[side].style.height)||0));
      const target=Math.max(...natural);
      pair.spacers.forEach((spacer,side)=>{
        const height=Math.round(Math.max(0,target-natural[side])*100)/100;
        if (Math.abs((parseFloat(spacer.style.height)||0)-height)>.5) {
          spacer.style.height=height+'px';changed=true;
        }
      });
    });
    if ((changed || restorePosition) && position) {
      const y = positionAt(right, position);
      expected[0]=y; expected[1]=y;
      frames.forEach(frame => frame.contentWindow.scrollTo({top:y,behavior:'instant'}));
    }
  }
  const scheduleAlignment = () => {
    clearTimeout(alignTimer);
    alignTimer = setTimeout(() => {cancelAnimationFrame(alignFrame);alignFrame=requestAnimationFrame(align);},100);
  };
  function connect() {
    const docs = frames.map(frame => frame.contentDocument);
    const order = getAnchors(docs[0]).map(anchorKey);
    let rightIndex = -1;
    const rightOrder = getAnchors(docs[1]).map(anchorKey);
    const ids = order.filter(id => {
      const index = rightOrder.indexOf(id);
      if (index<0 || index<=rightIndex) return false;
      rightIndex=index; return true;
    });
    pairs = ids.map(id => ({id,elements:docs.map(doc=>getAnchors(doc).find(element=>anchorKey(element)===id))}));
    const ends = docs.map(doc => {
      const node = doc.createElement('div');node.dataset.comparisonEnd='';doc.body.append(node);return node;
    });
    pairs.push({id:'__end',elements:ends});
    pairs.forEach(pair => {
      pair.spacers = pair.elements.map(element => {
        const spacer = element.ownerDocument.createElement('div');
        spacer.className='np-comparison-spacer';spacer.setAttribute('aria-hidden','true');
        spacer.style.cssText='display:block;height:0;margin:0;padding:0;border:0;overflow:hidden;pointer-events:none;flex:0 0 auto';
        element.before(spacer);return spacer;
      });
    });
    frames.forEach((frame,index) => {
      const win = frame.contentWindow, doc = win.document;
      const style = doc.createElement('style');
      style.textContent='html{scroll-behavior:auto!important;overflow-anchor:none}body{overflow-x:hidden;overflow-anchor:none}section[id],header#head{scroll-margin-top:var(--np-comparison-inset,90px)!important}.np-comparison-spacer{visibility:hidden}';
      doc.head.append(style);
      win.addEventListener('scroll',()=>sync(index),{passive:true});
      win.addEventListener('resize',scheduleAlignment);
      doc.addEventListener('toggle',scheduleAlignment,true);
      doc.addEventListener('change',scheduleAlignment,true);
      doc.addEventListener('click',scheduleAlignment,true);
      doc.addEventListener('keydown',event=>{if(event.key==='Escape'){event.preventDefault();hide();}},true);
      const observer = new ResizeObserver(scheduleAlignment);
      const main = doc.querySelector('.wrap') || doc.body;
      observer.observe(main);
      if (doc.fonts && doc.fonts.ready) doc.fonts.ready.then(scheduleAlignment);
      frame.__comparisonObserver = observer;
    });
    ready=true;
    align();
    const y=positionAt(frames[1].contentWindow,savedPosition);
    expected[0]=y;expected[1]=y;
    frames.forEach(frame=>frame.contentWindow.scrollTo({top:y,behavior:'instant'}));
    shell.dataset.ready='true';
  }
  function loadFrames() {
    const loaded = [false, false];
    [reference,location.href].forEach((url,index) => {
      const frame = make('iframe', {title:index===0?'Published report \u00b7 '+neuron:'Replication report \u00b7 '+neuron, src:url});
      frames[index]=frame;
      frame.addEventListener('load',()=>{
        const doc = frame.contentDocument;
        // Hosted reports first load a small gzip loader, then replace its document.
        // Bind to the completed report, not the loader's first load event.
        if (loaded[index] || !doc?.getElementById('head') || !doc.getElementById('main')) return;
        loaded[index]=true;
        panes[index].classList.add('np-compare-loaded');
        if (loaded.every(Boolean)) connect();
      });
      panes[index].prepend(frame);
    });
  }
  function show() {
    if (opened) return;
    savedPosition=capture(window);opened=true;shell.hidden=false;
    document.documentElement.classList.add('np-comparing');
    setToggleState(true);
    if (!frames.length) loadFrames();
    else if (ready) {
      align();
      const y=positionAt(frames[1].contentWindow,savedPosition);
      expected[0]=y;expected[1]=y;
      frames.forEach(frame=>frame.contentWindow.scrollTo({top:y,behavior:'instant'}));
    }
    close.focus({preventScroll:true});
  }
  function hide() {
    if (!opened) return;
    const position=ready?capture(frames[1].contentWindow):savedPosition;
    opened=false;shell.hidden=true;document.documentElement.classList.remove('np-comparing');
    setToggleState(false);
    window.scrollTo({top:positionAt(window,position),behavior:'instant'});
    toggle.focus({preventScroll:true});
  }
  toggle.addEventListener('click',()=>opened?hide():show());
  close.addEventListener('click',hide);
  document.addEventListener('keydown',event=>{if(opened && event.key==='Escape'){event.preventDefault();hide();}});
  let dragging=false;
  const resizeAt = ratio => {
    if (ready && !pendingPosition) pendingPosition=capture(frames[1].contentWindow);
    const value=Math.max(25,Math.min(75,ratio));
    shell.style.setProperty('--split',value+'%');divider.setAttribute('aria-valuenow',String(Math.round(value)));
    scheduleAlignment();
  };
  divider.addEventListener('pointerdown',event=>{dragging=true;divider.setPointerCapture(event.pointerId);shell.classList.add('np-compare-dragging');});
  divider.addEventListener('pointermove',event=>{if(dragging)resizeAt(event.clientX/window.innerWidth*100);});
  const stopDrag=()=>{dragging=false;shell.classList.remove('np-compare-dragging');scheduleAlignment();};
  divider.addEventListener('pointerup',stopDrag);divider.addEventListener('pointercancel',stopDrag);
  divider.addEventListener('keydown',event=>{
    if(event.key==='ArrowLeft'||event.key==='ArrowRight'){event.preventDefault();resizeAt(Number(divider.getAttribute('aria-valuenow'))+(event.key==='ArrowLeft'?-2:2));}
    if(event.key==='Home'){event.preventDefault();resizeAt(50);}
  });
  if (new URLSearchParams(location.search).get('compare') !== '0') {
    if (document.readyState === 'complete') show();
    else window.addEventListener('load',show,{once:true});
  }
})();
