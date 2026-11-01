const $=(q,root=document)=>root.querySelector(q);
const esc=v=>String(v).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt=v=>new Intl.NumberFormat('en-US',{maximumFractionDigits:1}).format(v);
const variants=[['positive','Positive'],['path_positive','Path-guided +'],['diversity','Diversity'],['negative','Negative'],['path_negative','Path-guided −']];
const cache=new Map(),heatCache=new Map(),bundleCache=new Map();
const optimizedCache=new Map();
let publication=null,catalogue={},examplePublication=null;
let data,refs,ours,state,showHeat=false,lastTrigger;
const key=()=>`${state.layer}:${state.channel}`;
const layer=()=>data.layers.find(l=>l.name===state.layer);
const current=()=>cache.get(key())?.value;
const defaultStep=()=>data.defaultSteps||1024;
const slotInfo=(own,slot)=>state.step===defaultStep()?own?.variantInfo?.[slot]:(own?.legacyVariantInfo?.[slot]||own?.variantInfo?.[slot]);
const reference=slot=>refs.neurons[key()].flags[variants.findIndex(v=>v[0]===slot)]==='1'?(ours[key()]?.reference?.[slot]||`${data.referenceBase}fv/${slot}/${state.layer}/${state.channel}.png`):null;
const allSamples=()=>current()?.exampleBuckets.flatMap(b=>b.examples)||[];
async function unpack(response){
  const buffer=await response.arrayBuffer(),bytes=new Uint8Array(buffer);
  if(bytes[0]!==31||bytes[1]!==139)return buffer;
  return new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
}
async function json(path){const r=await fetch(path);if(!r.ok)throw Error(`${r.status} · ${path}`);return JSON.parse(new TextDecoder().decode(await unpack(r)));}
function readURL(){
  const p=new URLSearchParams(location.search),l=data.layers.find(l=>l.name===p.get('layer'))||data.layers.find(l=>l.name==='mixed4c');
  const raw=Number(p.get('neuron')??(l.name==='mixed4c'?477:0));
  state={layer:l.name,channel:Number.isInteger(raw)&&raw>=0&&raw<l.channels?raw:0,
    variant:variants.some(v=>v[0]===p.get('variant'))?p.get('variant'):'positive',
    compare:p.get('compare')!=='0',step:Number(p.get('step')||data.defaultSteps||1024)};
}
function syncURL(){const p=new URLSearchParams({layer:state.layer,neuron:state.channel,compare:state.compare?'1':'0'});if(state.variant!=='positive')p.set('variant',state.variant);if(state.step!==data.defaultSteps)p.set('step',state.step);history.replaceState(null,'',`?${p}`);}
function selectNeuron(l,ch){state.layer=l;state.channel=Number(ch);state.step=data.defaultSteps;syncURL();renderNavigation();renderDetail();loadSamples();loadOptimized();}
function mergeOptimized(incoming){
  const id=`${incoming.layer}:${incoming.channel}`,previous=ours[id]||{},results={};
  for(const [slot] of variants){
    results[slot]={...previous.ours?.[slot]};delete results[slot][defaultStep()];
    Object.assign(results[slot],incoming.ours?.[slot]||{});
  }
  ours[id]={...previous,...incoming,ours:results,legacyVariantInfo:previous.legacyVariantInfo||previous.variantInfo||{},reference:previous.reference};
}
async function loadOptimized(){
  if(!publication||!catalogue[key()])return;
  const id=key(),release=publication.release;
  if(!optimizedCache.has(id))optimizedCache.set(id,json(`/api/neuron/${release}/${state.layer}/${state.channel}`).then(mergeOptimized).catch(error=>{optimizedCache.delete(id);throw error;}));
  try{await optimizedCache.get(id);if(key()===id)renderDetail();}catch{if(key()===id){$('#toast').textContent='优化结果加载失败，切换 neuron 后可重试';$('#toast').classList.add('visible');setTimeout(()=>$('#toast').classList.remove('visible'),4000);}}
}
async function loadPublished(){
  if(!data.publishedApi)return;
  try{
    const active=await json('/api/published');if(!active.release)return;
    const index=await json(`/api/catalog/${active.release}`);
    publication=active;catalogue=index.nodes;renderNavigation();await loadOptimized();
  }catch{/* Keep the verified bundled snapshot available if the storage service is unavailable. */}
}
async function loadExamplePublication(){
  if(!data.publishedApi)return;
  try{const active=await json('/api/examples/published');if(active.release)examplePublication=active;}catch{}
}
function hasPublishedExamples(id=key()){return !!examplePublication&&(!examplePublication.nodes||examplePublication.nodes.includes(id));}
function renderNavigation(){
  let group='';$('#layers').innerHTML=data.layers.map(l=>{const h=group===l.group?'':`<div class="layer-group">${esc(l.group)}</div>`;group=l.group;return `${h}<button class="layer-button ${l.name===state.layer?'active':''}" data-layer="${l.name}" aria-pressed="${l.name===state.layer}">${l.name}<span>${l.channels}</span></button>`;}).join('');
  $('#layer-name').textContent=state.layer;$('#channel-count').textContent=layer().channels;
  $('#neuron-input').max=layer().channels-1;$('#neuron-input').value='';$('#jump-error').textContent='';
  $('#channel-grid').innerHTML=Array.from({length:layer().channels},(_,ch)=>`<button class="${ch===state.channel?'selected':''} ${ours[`${state.layer}:${ch}`]||catalogue[`${state.layer}:${ch}`]?.includes('1')?'populated':''}" data-channel="${ch}" aria-label="${state.layer}:${ch}" aria-pressed="${ch===state.channel}">${ch}</button>`).join('');
}
function imageButton(path,title){return `<button class="image-button" data-image="${path}" data-title="${esc(title)}" aria-label="放大 ${esc(title)}"><img src="${path}" alt="${esc(title)}" decoding="async"></button>`;}
function empty(text){return `<div class="empty-image"><span>—</span><strong>${text}</strong></div>`;}
function visualStage(){
  const id=state.variant,name=variants.find(v=>v[0]===id)[1],own=ours[key()],result=own?.ours?.[id];
  let info=slotInfo(own,id);
  const steps=result?Object.keys(result).map(Number).sort((a,b)=>a-b):[];
  if(own?.pending?.[id]&&!steps.includes(defaultStep()))steps.push(defaultStep());
  steps.sort((a,b)=>a-b);
  if(steps.length&&!result?.[state.step]&&!own?.pending?.[id])state.step=steps.at(-1);
  const image=result?.[state.step],pending=own?.pending?.[id]==='quality_review';
  info=slotInfo(own,id);
  const ref=reference(id),status=refs.neurons[key()].flags[variants.findIndex(v=>v[0]===id)];
  const official=`https://already-clean.goodfire.com/microscope/inceptionv1.html?layer=${state.layer}&neuron=${state.channel}`;
  const refCard=`<article class="visual-card"><div class="visual-card-heading"><span>Nick / Goodfire</span><a href="${official}" target="_blank" rel="noopener" aria-label="官网原始页面">↗</a></div>${ref?imageButton(ref,`Nick · ${key()} · ${name}`):empty(status==='h'?'官网未公开':'暂无图片')}</article>`;
  const score=info?.scores?.[state.step];
  const ourCard=`<article class="visual-card ours"><div class="visual-card-heading"><span>Ours</span><span class="tag teal">${image?`${state.step} steps`:pending?'待修复':'未优化'}</span></div>${image?imageButton(image,`Ours · ${key()} · ${name} · ${state.step}`):empty(pending?'待修复':'待优化')}<div class="visual-caption"><span>${info?`${info.config.size} × ${info.config.size} · seed ${info.config.seed}${(info.members||info.batchSize)>1?` · ${info.members||info.batchSize} images`:''}`:''}</span><span>${score?`Act. ${fmt(score.reduce((a,b)=>a+b,0)/score.length)}`:''}</span></div></article>`;
  return `<div class="variant-tabs" role="group" aria-label="解释类型">${variants.map(([id,label])=>`<button class="variant-tab ${id===state.variant?'active':''}" data-variant="${id}" aria-pressed="${id===state.variant}">${reference(id)?`<img src="${reference(id)}" alt="${label}" decoding="async">`:'<span class="variant-missing">—</span>'}<strong>${label}</strong></button>`).join('')}</div><div class="visual-stage ${state.compare?'comparing':'single'}">${refCard}${state.compare?ourCard:''}</div>${state.compare&&steps.length>1?`<div class="step-bar"><span class="step-title">STEP</span><div class="step-buttons">${steps.map(s=>`<button data-step="${s}" class="${s===state.step?'active':''}" aria-pressed="${s===state.step}">${s===0?'init':s}</button>`).join('')}</div></div>`:''}`;
}
function sampleVisual(s){
  const image=s.atlasUrl||`${data.sampleRoot}atlases/${s.atlas}.webp`;
  const [h,w]=s.heatShape,xy=s.mapLocation||s.location;
  const columns=s.atlasColumns||8,rows=s.atlasRows||1;
  const px=columns>1?s.tile%columns*100/(columns-1):0,py=rows>1?Math.floor(s.tile/columns)*100/(rows-1):0;
  const point=s.activationPosition||[(xy[0]+.5)/h,(xy[1]+.5)/w];
  return `<div class="natural-crop atlas-crop" role="img" aria-label="${esc(s.image)}" style="background-image:url('${image}');background-size:${columns*100}% ${rows*100}%;background-position:${px}% ${py}%"><canvas class="activation-map" data-heat="${s.id}" aria-hidden="true" hidden></canvas>${s.heatPeak>0?`<span class="activation-point" style="left:${point[1]*100}%;top:${point[0]*100}%" ${showHeat?'':'hidden'}></span>`:''}<span class="sample-rank">${s.pole==='negative'?'MIN ':''}#${s.rank}</span></div>`;
}
function heatToggle(){return `<button class="heat-toggle ${showHeat?'active':''}" data-heat-toggle aria-pressed="${showHeat}">${showHeat?'隐藏':'显示'}激活位置</button>`;}
function grid(bucket){return `<div class="samples-grid">${bucket.examples.map(s=>`<button class="sample-button" data-sample="${s.id}" aria-label="第 ${s.rank} 名 ${s.label} 激活 ${fmt(s.activation)}">${sampleVisual(s)}<span class="sample-label">${s.label}</span><span class="sample-score">${fmt(s.activation)}</span></button>`).join('')}</div>`;}
function samples(){
  if(!data.sampleRoot&&!hasPublishedExamples())return '<section class="sample-loading">尚未导入自然样例扫描结果</section>';
  const n=current();
  if(!n)return `<section class="sample-loading"><span>${cache.get(key())?.error?'样例加载失败':'正在加载样例…'}</span>${cache.get(key())?.error?'<button id="retry-samples">重试</button>':''}</section>`;
  return `<section class="sample-section"><div class="section-head"><h2>Top-activating images</h2>${heatToggle()}</div>${grid(n.exampleBuckets[0])}</section><section class="sample-section"><div class="section-head"><h2>Additional Examples</h2><span class="dataset-badge">OUR SCAN · ${(data.dataset?.samples??50000).toLocaleString()}</span></div>${n.exampleBuckets.slice(1).map(b=>`<div class="example-bucket"><div class="bucket-heading"><h3>${esc(b.title||({'p9998':'99.98%','p998':'99.8%','p999':'99.9%','p99':'99%'}[b.id]||'Most negative'))}</h3><span>#${b.examples[0].rank}–${b.examples.at(-1).rank}</span></div>${grid(b)}</div>`).join('')}</section>`;
}
function details(){
  const own=ours[key()],info=slotInfo(own,state.variant),crop=current()?.cropPolicy;
  const resultSource=publication&&catalogue[key()]?`/api/neuron/${publication.release}/${state.layer}/${state.channel}`:'our-results.json';
  return `<details class="provenance"><summary>数据与参数 <span>＋</span></summary><div class="provenance-grid"><div><h3>样例</h3><p>${data.dataset?`${(data.dataset?.samples??50000).toLocaleString()} 张真实图片，224 × 224 输入；正向按空间最大 post-ReLU、负向按空间最小 pre-ReLU 排名。`:'尚未导入样例。'} ${crop?'按分支感受野裁剪，上限 224 × 224；边界处平移裁剪框，热图与极值位置同步对齐。':''} 热图是逐图归一化的层响应，不是像素归因。</p>${data.sampleRoot?`<p><a href="${data.sampleRoot}protocol.json" target="_blank">扫描协议 ↗</a></p>`:''}</div><div><h3>优化</h3>${info?`<p>${esc(info.activation)} · Gaussian σ ${info.config.gaussian_sigma} · seed ${info.config.seed} · ${info.config.size} × ${info.config.size} · ${info.config.steps} steps</p><p>Run: ${esc(info.runId)}</p><p>Model: ${esc(own.model.source_commit)}<br>Weights: ${esc(own.model.weights_sha256)}</p>`:'<p>尚未优化。</p>'}<p><a href="${resultSource}" target="_blank">完整配置 ↗</a> · <a href="https://already-clean.goodfire.com/" target="_blank" rel="noopener">Nick / Goodfire ↗</a></p></div></div></details>`;
}
function renderDetail(){
  document.title=`${key()} · InceptionV1`;
  $('#detail').innerHTML=`<div class="detail-top"><div><div class="breadcrumb">INCEPTIONV1</div><h1>${state.layer} <span>: ${state.channel}</span></h1></div><div class="top-actions"><button class="icon-button" id="prev-neuron" aria-label="上一个 neuron">‹</button><button class="icon-button" id="next-neuron" aria-label="下一个 neuron">›</button><button id="compare-toggle" class="compare-button ${state.compare?'active':''}" aria-pressed="${state.compare}">◫ Compare</button></div></div><div class="compact-index"><label for="compact-channel">Neuron</label><input id="compact-channel" type="number" min="0" max="${layer().channels-1}" value="${state.channel}"></div>${visualStage()}<div id="sample-content">${samples()}</div>${details()}`;
  paintHeatmaps();
}
async function loadSamples(retry=false){
  if(!data.sampleRoot&&!hasPublishedExamples())return;
  const id=key(),channel=state.channel,bundle=`${state.layer}/${Math.floor(channel/16)}`;
  if(retry){cache.delete(id);heatCache.delete(id);bundleCache.delete(bundle);}
  if(!cache.has(id)){
    const entry={};cache.set(id,entry);
    const path=hasPublishedExamples(id)?`/api/examples/${examplePublication.release}/${state.layer}/${channel}`:data.rfExamples?.nodes.includes(id)?`${data.rfExamples.nodeRoot}${state.layer}/${channel}.json`:`${data.sampleRoot}neurons/${state.layer}/${channel}.json`;
    entry.promise=json(path).then(value=>{entry.value=value;}).catch(error=>{entry.error=error.message;});
  }
  await cache.get(id).promise;
  if(key()!==id)return;
  renderDetail();
}
async function paintHeatmaps(){
  document.querySelectorAll('.activation-point').forEach(x=>x.hidden=!showHeat);
  document.querySelectorAll('canvas[data-heat]').forEach(x=>x.hidden=!showHeat);
  const n=current();if(!showHeat||!n)return;
  const id=key();
  if(!heatCache.has(id))heatCache.set(id,n.heatBase64?Promise.resolve(Uint8Array.from(atob(n.heatBase64),c=>c.charCodeAt(0))):fetch(`${data.sampleRoot}${n.heatPath}`).then(async r=>{if(!r.ok)throw Error('Heatmap unavailable');return new Uint8Array(await r.arrayBuffer());}));
  let bytes;try{bytes=await heatCache.get(id);}catch{heatCache.delete(id);$('#toast').textContent='热图加载失败，请重试';$('#toast').classList.add('visible');setTimeout(()=>$('#toast').classList.remove('visible'),3000);return;}
  if(key()!==id||!showHeat)return;
  const byId=new Map(allSamples().map(s=>[s.id,s]));
  document.querySelectorAll('canvas[data-heat]').forEach(canvas=>{
    const s=byId.get(canvas.dataset.heat);if(!s)return;
    const [h,w]=s.heatShape;canvas.width=w;canvas.height=h;
    const ctx=canvas.getContext('2d'),pixels=ctx.createImageData(w,h);
    for(let i=0;i<w*h;i++){const t=bytes[s.heatOffset+i]/255,j=i*4,neg=s.pole==='negative';pixels.data[j]=neg?40:255;pixels.data[j+1]=neg?140:Math.round(185-100*t);pixels.data[j+2]=neg?255:25;pixels.data[j+3]=Math.round(200*t);}
    ctx.putImageData(pixels,0,0);canvas.hidden=false;
  });
}
function showModal(title,content,caption,trigger){lastTrigger=trigger;$('#lightbox-title').textContent=title;$('#lightbox-image').innerHTML=content;$('#lightbox-caption').textContent=caption;$('#lightbox').showModal();}
document.addEventListener('click',e=>{
  const b=e.target.closest('button');if(!b||!data)return;
  if(b.dataset.layer){selectNeuron(b.dataset.layer,0);return;}
  if(b.dataset.channel!==undefined){selectNeuron(state.layer,b.dataset.channel);return;}
  if(b.dataset.variant){state.variant=b.dataset.variant;state.step=data.defaultSteps;syncURL();renderDetail();return;}
  if(b.dataset.step){state.step=Number(b.dataset.step);syncURL();renderDetail();return;}
  if(b.id==='compare-toggle'){state.compare=!state.compare;syncURL();renderDetail();return;}
  if(b.id==='prev-neuron'||b.id==='next-neuron'){selectNeuron(state.layer,(state.channel+(b.id==='next-neuron'?1:-1)+layer().channels)%layer().channels);return;}
  if(b.id==='retry-samples'){loadSamples(true);return;}
  if(b.dataset.heatToggle!==undefined){showHeat=!showHeat;document.querySelectorAll('[data-heat-toggle]').forEach(t=>{t.classList.toggle('active',showHeat);t.setAttribute('aria-pressed',showHeat);t.textContent=`${showHeat?'隐藏':'显示'}激活位置`;});paintHeatmaps();return;}
  if(b.dataset.image){showModal(b.dataset.title,`<img src="${b.dataset.image}" alt="${esc(b.dataset.title)}">`,'',b);return;}
  if(b.dataset.sample!==undefined){const s=allSamples().find(s=>s.id===b.dataset.sample);if(s){showModal(`${s.label} · #${s.rank}`,`<div class="sample-dialog">${sampleVisual(s)}${heatToggle()}</div>`,`${s.image} · ${s.pole==='negative'?'min pre-ReLU':'max post-ReLU'} ${fmt(s.activation)} · [${s.location.join(', ')}]`,b);paintHeatmaps();}return;}
  if(b.id==='close-lightbox')$('#lightbox').close();
});
document.addEventListener('error',e=>{if(e.target.tagName==='IMG'){e.target.closest('.image-button')?.replaceWith(Object.assign(document.createElement('div'),{className:'empty-image',textContent:'图片加载失败'}));}},true);
$('#lightbox').addEventListener('click',e=>{if(e.target===$('#lightbox'))$('#lightbox').close();});
$('#lightbox').addEventListener('close',()=>{$('#lightbox-image').replaceChildren();lastTrigger?.focus();});
$('#jump-form').addEventListener('submit',e=>{e.preventDefault();const ch=Number($('#neuron-input').value);if(!Number.isInteger(ch)||ch<0||ch>=layer().channels){$('#jump-error').textContent=`0–${layer().channels-1}`;return;}selectNeuron(state.layer,ch);});
document.addEventListener('change',e=>{if(e.target.id==='compact-channel'){const n=Number(e.target.value);if(Number.isInteger(n)&&n>=0&&n<layer().channels)selectNeuron(state.layer,n);}});
window.addEventListener('popstate',()=>{if(data){readURL();renderNavigation();renderDetail();loadSamples();loadOptimized();}});
try{data=await json('data.json');[refs,ours]=await Promise.all([json(data.referenceCatalog),json(data.ours),loadExamplePublication()]);readURL();renderNavigation();renderDetail();loadSamples();loadPublished();}
catch(e){$('#detail').innerHTML=`<div class="empty-page"><h2>加载失败</h2><p>${esc(e.message)}</p><button onclick="location.reload()">重试</button></div>`;}
