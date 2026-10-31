(() => {
'use strict';
const { language, translate: L, updateLanguageLinks } = window.NeuronDemoI18n;
const $=(s,p=document)=>p.querySelector(s);
const esc=x=>String(L(x)??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const cache=new Map();
async function get(path){if(cache.has(path))return cache.get(path);const promise=fetch(path).then(r=>{if(!r.ok)throw Error(L('读取记录失败：')+path);return r.json();});cache.set(path,promise);return promise;}
const text=x=>typeof x==='string'?x:JSON.stringify(x,null,2);
const fmt=n=>Number(n).toLocaleString(language==='zh'?'zh-CN':'en-US');
function time(v,short=false){if(!v)return '—';return new Date(v).toLocaleString(language==='zh'?'zh-CN':'en-US',{timeZone:'Asia/Shanghai',month:short?undefined:'2-digit',day:short?undefined:'2-digit',hour:'2-digit',minute:'2-digit',second:'2-digit',hour12:false});}
function dur(v){if(v==null)return '—';const n=Math.round(v);return n<60?n+(language==='zh'?'秒':' s'):Math.floor(n/60)+(language==='zh'?'分':' min ')+(n%60? n%60+(language==='zh'?'秒':' s'):'');}
function node(tag,cls,content){const n=document.createElement(tag);if(cls)n.className=cls;if(content!=null)n.textContent=L(content);return n;}
function link(label,path){const a=node('a','',L(label)+' ↗');a.href=path;a.target='_blank';a.rel='noopener';return a;}
function paragraph(host,t,cls=''){host.append(node('p',cls,t));}
function outputLink(host,label,path){const d=node('div','output-links');d.append(link(label,path));host.append(d);}
function code(host,title,value,dark=false){
 const all=text(value),wrap=node('div','code-card'+(dark?' dark':'')),bar=node('div','code-title'),pre=node('pre');
 bar.append(node('span','',title));const copy=node('button','',L('复制'));copy.onclick=async()=>{try{await navigator.clipboard.writeText(all);copy.textContent=L('已复制');}catch{copy.textContent=L('请选中文本复制');}};const acts=node('span','code-actions'),read=node('button','',L('放大阅读'));read.onclick=()=>openTextModal(title,all);acts.append(read,copy);bar.append(acts);wrap.append(bar);
 let open=all.length<=4800;const draw=()=>{pre.textContent=open?all:all.slice(0,4800)+'\n\n……';};draw();wrap.append(pre);
 if(all.length>4800){const b=node('button','code-more',L('展开完整内容（')+fmt(all.length)+L('个字符）'));b.onclick=()=>{open=!open;draw();b.textContent=open?L('收起预览'):L('展开完整内容（')+fmt(all.length)+L('个字符）');};wrap.append(b);}host.append(wrap);return wrap;
}
const steps=[
{id:"scan",group:L("准备"),nav:L("准备证据"),role:L("固定 CPU 程序"),title:L("读取已有激活，抽取 500 个位置"),summary:L("程序整理激活统计、自然上下文和 token 排名，再从频谱各区间抽取待打标的样本。"),snapshot:"initial",story:[[L("输入"),L("Qwen3-32B 在 Diverse 100M 语料上的 gate、up 和 product 激活记录。")],[L("操作"),L("按各频谱区间语料计数的 log10 值分配 500 点配额，再在区间内均匀抽样。")],[L("产物"),L("初始证据 HTML、结构化统计，以及待打标的 500 个位置。")]],output:"scan",facts:[]},
{id:"agent1",group:L("Stage 1 · 初步解释"),nav:L("解释与分桶"),role:L("Agent 1"),title:L("根据实测证据提出解释"),summary:L("Agent 1 阅读正负极样本、分位上下文、token 选择性及输出效果，提交解释和响应规则。"),snapshot:"agent1",story:[[L("输入"),L("激活分布、两极强响应、较弱样本及其原始上下文。")],[L("操作"),L("归纳响应对应的语义条件，定义最多 3 个正极桶、3 个负极桶，以及一个 No visible cue。")],[L("产物"),L("Colorado local-context 初步解释，以及 2 个正极桶、3 个负极桶和 No visible cue。")]],trace:"agent1",output:"01-interpretation",facts:[]},
{id:"feedback",group:L("Stage 1 · 初步解释"),nav:L("样本打标"),role:L("固定程序＋预测 API"),title:L("根据解释逐样本预测"),summary:L("宿主程序冻结提交的解释，通过两路并行请求将样本交给独立预测器。"),snapshot:"agent1",story:[[L("输入"),L("固定解释、响应规则，以及每个位置的完整前缀和目标 token。")],[L("操作"),L("预测桶、响应方向和理由。本次两路分别处理 256 和 244 个位置，预测结果随后与实测值关联。")],[L("产物"),L("预测反馈，以及使用同一批 495 个已打标位置绘制的 Spectrum 和 Inside the neuron。")]],aux:"initial-prediction",output:"prediction-1",facts:[]},
{id:"agent2",group:L("Stage 2 · 实验"),nav:L("设计并测量实验"),role:L("Agent 2"),title:L("用对照实验检验解释"),summary:L("Agent 2 阅读初步解释及预测反馈，设计匹配输入，调用宿主工具测量响应和相关回路。"),snapshot:"agent2",story:[[L("输入"),L("初步解释、样本预测，以及已有激活证据。")],[L("操作"),L("比较 Colorado 城市和电话区号，检查青海与代码样本，测量替换实验、搜索、干预和 RelP 归因图。")],[L("产物"),L("实测对照、归因图、语义 Clusters、Family 轨迹和独立文本测试。")]],trace:"agent2",output:"02-experiments",facts:[]},
{id:"agent3",group:L("Stage 3 · 修订"),nav:L("修订解释"),role:L("Agent 3"),title:L("结合反馈与实验修订解释"),summary:L("Agent 3 阅读实验结果与反例，撰写最终解释、Criticism 和图注。"),snapshot:"agent3",story:[[L("输入"),L("初步解释、预测错误、对照测量和回路证据。")],[L("操作"),L("评估 Colorado 联系信息模式和多种负响应，修订解释文字及图表结论。")],[L("产物"),L("Colorado local-contact 最终解释、Criticism 和报告注释。")]],trace:"agent3",output:"03-revision",facts:[]},
{id:"publish",group:L("报告"),nav:L("评阅与发布"),role:L("固定程序＋评阅 API"),title:L("评阅解释，生成报告"),summary:L("独立模型评阅报告，固定程序将解释、实测图表、评阅及费用记录合并成可交互的 HTML。"),snapshot:"published",story:[[L("输入"),L("最终提交、测量结果、图表注释和 API 用量记录。")],[L("操作"),L("执行报告评阅、组装 HTML，并接入原博客的双栏比较。")],[L("产物"),L("完整的交互式 neuron report，包含实验依据、评阅和解释成本。")]],aux:"reader",output:"publish",facts:[]}
];
let data,current=0,activeTab='overview',version=0;
const tabNames={overview:L('过程讲解'),prompt:L('实际Prompt'),actions:L('工具轨迹'),output:L('阶段产物')};
const titles={headline:L('顶部主图'),custom:L('定制实验图'),graphs_curated:'Three graphs',spectrum:'Activation spectrum',clusters:L('语义Clusters'),routes:L('原始Routes'),anatomy:'Inside the neuron',family:'Where it sits',circuit:'Circuit',tests:L('Tests字段'),criticism:'Criticism',agent_read_full:L('独立整体评阅'),interpretation_costs:'Interpretation Costs'};
function snapshot(id){const saved=data.snapshots.find(s=>s.id===id);return {...saved,title:L(({initial:'初始证据',agent1:'Stage 1报告',agent2:'Stage 2报告',agent3:'Stage 3报告',published:'发布报告'})[id])};}
function traceMeta(id){return data.traces.find(t=>t.id===id);}
function auxFor(s){return data.aux_calls.filter(a=>a.kind===s.aux);}
function buildNav(){const nav=$('#step-nav');let last='';steps.forEach((s,i)=>{
 if(s.group!==last){nav.append(node('div','nav-group',s.group));last=s.group;}
 const b=node('button','step-button'+(s.branch?' branch':''));b.dataset.index=i;b.innerHTML='<span class="num">'+String(i+1).padStart(2,'0')+'</span><span>'+esc(s.nav)+(s.sub?'<small>'+esc(s.sub)+'</small>':'')+'</span>';b.onclick=()=>select(i);nav.append(b);
});}
function select(i,tab){current=Math.max(0,Math.min(steps.length-1,i));activeTab=tab||'overview';version++;const s=steps[current];if(!document.body.classList.contains('cover-open'))history.replaceState(null,'','#'+s.id);$('#step-counter').textContent=(current+1)+'／'+steps.length;document.querySelectorAll('.step-button').forEach((b,k)=>{b.classList.toggle('active',k===current);b.classList.toggle('past',k<current);b.setAttribute('aria-current',k===current?'step':'false');});
 const detail=$('#detail');detail.innerHTML='<div class="step-heading"><div class="step-eyebrow">'+esc(s.group)+'<span class="role">'+esc(s.role)+'</span></div><h2>'+esc(s.title)+'</h2><p>'+esc(s.summary)+'</p>'+(s.note?'<div class="branch-note">'+esc(s.note)+'</div>':'')+'</div><div class="tabs" role="tablist"></div><div class="tabbody"></div><div class="step-controls"></div>';
 const tabs=$('.tabs',detail);for(const [key,label] of Object.entries(tabNames)){const b=node('button',key===activeTab?'active':'',label);b.setAttribute('role','tab');b.setAttribute('aria-selected',key===activeTab?'true':'false');b.onclick=()=>changeTab(key);tabs.append(b);}
 const controls=$('.step-controls',detail),prev=node('button','',L('← 上一步')),next=node('button','primary',current===steps.length-1?L('已到最终页面'):L('下一步 →'));prev.disabled=!current;next.disabled=current===steps.length-1;prev.onclick=()=>select(current-1);next.onclick=()=>select(current+1);controls.append(prev,node('span','middle',String(current+1).padStart(2,'0')+'／'+steps.length),next);
 demoPreview(s);renderTab(s,version);syncDemoControls();updateLanguageLinks();if(!document.body.classList.contains('cover-open'))window.scrollTo({top:0,behavior:'auto'});
}
function changeTab(key){activeTab=key;version++;document.querySelectorAll('.tabs button').forEach((b,i)=>{const k=Object.keys(tabNames)[i];b.classList.toggle('active',k===key);b.setAttribute('aria-selected',k===key?'true':'false');});renderTab(steps[current],version);}
async function renderTab(s,v){const host=$('.tabbody');host.innerHTML='<div class="loading">'+esc(L('正在读取记录……'))+'</div>';try{
 if(activeTab==='overview'){host.innerHTML='';demoOverview(host,s);return;}
 if(activeTab==='prompt'){host.innerHTML='';if(s.trace){const t=await get('content/'+s.trace+'.json');if(v!==version)return;promptTrace(host,t);}else if(s.aux){await apiBrowser(host,s,v);}else{paragraph(host,L('这一段由固定程序执行，没有向LLM发出任务。下面保留它实际使用的参数或输入。'));code(host,L('执行记录'),s.id==='scan'?data.preparation:data.costs);}
 return;}
 if(activeTab==='actions'){host.innerHTML='';if(s.trace){const t=await get('content/'+s.trace+'.json');if(v!==version)return;actions(host,t);}else if(s.aux){await apiBrowser(host,s,v);}else fixedActions(host,s);return;}
 if(activeTab==='output'){host.innerHTML='';await outputs(host,s,v);}
 }catch(e){if(v===version){host.innerHTML='';paragraph(host,e.message);}}
}
function promptTrace(host,t){
 paragraph(host,L('按运行时的顺序展示任务、系统提示及agent读取的skill。这里保留阶段指令和关键操作节选。'));
 code(host,L('本阶段任务'),t.task,true);
 const sys=node('details','simple');sys.append(node('summary','',L('系统提示')));sys.ontoggle=()=>{if(sys.open&&sys.childElementCount===1)code(sys,L('系统提示'),t.systems.join('\n\n'));};host.append(sys);
 const select=node('select','doc-select');for(const [i,f] of t.skill_files.entries()){const opt=node('option','',f.name);opt.value=i;select.append(opt);}host.append(select);
 const target=node('div');host.append(target);const draw=()=>{target.innerHTML='';const f=t.skill_files[Number(select.value)||0];if(f)code(target,L('运行时的 ')+f.name,f.text);};select.onchange=draw;draw();
 const aux=null;
 if(aux){const d=node('details','simple');d.append(node('summary','',L('查看辅助任务的Prompt节选')));const pane=node('div');d.append(pane);let made=false;d.ontoggle=()=>{if(d.open&&!made){made=true;apiBrowser(pane,{aux},version);}};host.append(d);}
}
function actionName(c){const a=c.arguments,cmd=typeof a==='string'?a:(a?.command||text(a));
 if(/submit_interpretation/.test(cmd))return L('提交初步解释与响应规则');
 if(/complete_stage|complete_experiments|complete_revision|submit_revision|submit_experiments|complete\.py/.test(cmd))return L('提交本阶段结果');
 if(/run_experiments\.py/.test(cmd))return L('提交实验计划并等待测量');
 if(/run_tests\.py/.test(cmd))return L('运行独立文本测试');
 if(/label_spectrum\.py/.test(cmd))return L('请求语义分组打标');
 if(/SKILL\.md/.test(cmd))return L('读取本阶段skill');
 if(/references\//.test(cmd))return L('读取接口与流程说明');
 if(/read_evidence\.py/.test(cmd)){const matches=[...cmd.matchAll(/--section\s+([\w-]+)/g)].map(x=>x[1]);return L('读取证据')+(matches.length?(language==='zh'?'：':': ')+[...new Set(matches)].join(language==='zh'?'、':', '):L('与样本'));}
 if(/cat\s*>|open\(.+["']w["']|write_text/.test(cmd))return '编写计划、分析文件或修订稿';
 if(/python/.test(cmd))return L('读取结构化结果并做分析');
 if(/ls\s|find\s|pwd/.test(cmd))return L('检查文件与已有结果');
 return L('检查记录或读取文件');}
function actions(host,t){
 paragraph(host,L('下列操作选自保存的轨迹。命令保留实际用途，返回内容以结果摘要展示。'));
 host.append(node('p','trace-stats',L('当次运行共')+t.model_rounds+L('轮模型响应、')+t.tool_calls+L('次工具调用；下方展示')+t.calls.length+L('项关键操作。')));
 for(const [i,c] of t.calls.entries()){
  const d=node('details','trace-call'),summary=node('summary'),caption=node('span','call-caption');
  caption.append(node('span','',actionName(c)),node('small','',c.name+L(' · 模型第')+c.step+L('轮')));
  summary.append(node('span','call-no',String(i+1).padStart(2,'0')),caption);d.append(summary);
  let made=false;d.ontoggle=()=>{if(!d.open||made)return;made=true;code(d,L('命令节选'),c.arguments,true);paragraph(d,c.result_summary);};host.append(d);
 }
 host.append(link(L('操作节选JSON'),'content/'+t.id+'.json'));
}
function fixedActions(host,s){
 if(s.id==='scan'){code(host,L('采样记录'),data.preparation);return;}
 if(s.id==='feedback'){code(host,L('打标执行记录'),data.predictions);return;}
 code(host,'Interpretation Costs',data.costs);
}
async function apiBrowser(host,s,v){
 const calls=auxFor(s);paragraph(host,L('以下展示保存的Prompt与对应结果节选。涉及多样本任务时，保留完整的选中样本上下文。'));
 if(!calls.length){host.append(node('div','empty',L('本步骤没有Prompt示例。')));return;}
 for(const a of calls){const z=await get(a.file);if(v!==version)return;
  code(host,L('系统Prompt'),z.system,true);code(host,L('输入节选'),z.input_excerpt);code(host,L('对应结果节选'),z.output_excerpt);host.append(link(L('打开这个示例'),a.file));
 }
}
async function outputs(host,s,v){
 if(s.output==='scan'){code(host,L('采样记录'),data.preparation);outputLink(host,L('初始交互报告'),snapshot('initial').file);return;}
 if(s.output==='prediction-1'){const z=await get('content/prediction-results.json');if(v!==version)return;code(host,L('预测反馈'),z);return;}
 if(s.output==='publish'){const z=await get('content/prompt-reader.json');if(v!==version)return;code(host,L('独立评阅结果'),z.output_excerpt);code(host,'Interpretation Costs',data.costs);outputLink(host,L('完整交互报告'),snapshot('published').file);return;}
 const st=data.stages[s.output];if(!st)return;
 if(s.output==='03-revision'){const diff=node('div','diff');for(const [label,c]of [[L('初步解释'),data.stages['01-interpretation'].claim],[L('修订解释'),st.claim]]){const box=node('div','diff-block');box.append(node('small','',label),node('div','paper-title',c.title),node('div','',c.lede));diff.append(box);}host.append(diff);}
 code(host,L('提交的解释'),st.claim);
 if(s.output==='01-interpretation')code(host,'response_rules',st.response_rules);
 if(s.output==='02-experiments'){code(host,L('测量索引'),data.measurements);for(const name of ['request_round1.json','request_round2.json'])outputLink(host,name,'content/'+name);}
 outputLink(host,L('阶段提交'),st.output);
}
let modalRevision=0;
function openModal(title,kicker,mode){modalRevision++;const d=$('#reading-dialog');if(d.open)d.close();$('#reading-title').textContent=title;$('#reading-kicker').textContent=kicker;const c=$('#reading-content');c.replaceChildren();c.className='reading-content '+mode;$('#reading-actions').replaceChildren();d.showModal();return c;}
function openTextModal(title,all){const c=openModal(title,L('PROMPT · 运行记录'),'text-mode'),pre=node('pre','inspector-text',all);c.append(pre);const copy=node('button','',L('复制全文'));copy.onclick=async()=>{try{await navigator.clipboard.writeText(all);copy.textContent=L('已复制');}catch{copy.textContent=L('可选中文本复制');}};$('#reading-actions').append(copy);}
function openHTML(snap,section='head'){const c=openModal(snap.title,L('INTERACTIVE HTML · 保存的科学内容'),'iframe-mode');const loading=node('div','html-loading',L('正在载入保存的HTML……'));c.append(loading);const f=node('iframe');f.title=snap.title;f.src=snap.file+'#'+section;f.onload=()=>loading.remove();c.append(f);$('#reading-actions').append(link(L('单独打开'),snap.file+'#'+section));}
function openImage(src,title,snap,section='head'){const c=openModal(title,L('SAVED HTML · 实际页面截图'),'image-mode');const img=node('img');img.src=src;img.alt=title;c.append(img);if(snap){const b=node('button','',L('打开交互HTML'));b.onclick=()=>openHTML(snap,section);$('#reading-actions').append(b);}$('#reading-actions').append(link(L('查看图片原文件'),src));}
function previewItems(id){const h={id:'head',label:L('页面顶部'),file:'screenshots/'+id+'.png'};const list={initial:['distribution'],agent1:['anatomy'],agent2:['tests'],agent3:['criticism'],published:['spectrum']};const names={distribution:L('激活分布'),anatomy:'Inside the neuron',custom:L('实验图'),tests:'Tests',criticism:'Criticism',spectrum:'Activation spectrum'};return [h,...(list[id]||[]).map(k=>({id:k,label:names[k],file:'screenshots/'+id+'-'+k+'.png'}))];}
function demoPreview(s){const snap=snapshot(s.snapshot),host=$('#preview'),items=previewItems(s.snapshot);const defaults={scan:'distribution',agent1:'head',feedback:'anatomy',agent2:'tests',agent3:'head',publish:'spectrum'};let chosen=items.find(x=>x.id===defaults[s.id])||items[0];host.innerHTML='<div class="preview-card"><div class="preview-title"><span>'+esc(L('这时的HTML'))+'</span><span class="label">'+esc(snap.title)+'</span></div><div class="preview-tools"></div><button class="preview-picture" aria-label="'+esc(L('放大本步页面截图'))+'"><img class="preview-image" width="1164" height="655"></button><div class="preview-description"></div><div class="preview-actions"></div></div><p class="preview-note">'+(L('点击截图放大，或打开对应的交互HTML。'))+'</p>';
 const select=node('select');select.setAttribute('aria-label',L('选择预览版块'));items.forEach(x=>{const o=node('option','',x.label);o.value=x.id;select.append(o);});select.value=chosen.id;const zoom=node('button','',L('放大 ⤢'));$('.preview-tools',host).append(select,zoom);const img=$('img',host),open=node('button','',L('打开交互HTML')),standalone=link(L('单独打开'),snap.file);$('.preview-actions',host).append(open,standalone);const draw=()=>{img.src=chosen.file;img.alt=snap.title+' · '+chosen.label;img.onerror=()=>{if(chosen.id!=='head'){img.src='screenshots/'+s.snapshot+'.png';img.onerror=null;}};$('.preview-description',host).textContent=chosen.label+' · '+(chosen.id==='head'?L('对应阶段的页面顶部'):L('对应阶段的版块截图'));standalone.href=snap.file+'#'+chosen.id;};select.onchange=()=>{chosen=items.find(x=>x.id===select.value);draw();};zoom.onclick=()=>openImage(chosen.file,snap.title+' · '+chosen.label,snap,chosen.id);$('.preview-picture',host).onclick=zoom.onclick;open.onclick=()=>openHTML(snap,chosen.id);draw();}
function syncDemoControls(){if(!$('#demo-next'))return;const s=steps[current];$('#playback-caption').textContent=String(current+1).padStart(2,'0')+'／'+steps.length+' · '+s.group;const p=$('#playback-progress');p.replaceChildren();steps.forEach((x,i)=>{const b=node('button',i===current?'active':i<current?'past':'');b.setAttribute('aria-label',L('跳到第')+(i+1)+L('步：')+x.nav);b.onclick=()=>startDemo(i);p.append(b);});$('#demo-prev').disabled=current===0;$('#demo-next').innerHTML='<span>'+esc(current===steps.length-1?L('从头再看'):L('下一步：')+steps[current+1].nav)+'</span><span>→</span>';}
function startDemo(i=0){document.body.classList.remove('cover-open');select(i);window.scrollTo({top:0,behavior:'auto'});}
function showCover(){document.body.classList.add('cover-open');history.replaceState(null,'',location.pathname+location.search);updateLanguageLinks();window.scrollTo({top:0,behavior:'auto'});}
function aboutDemo(){const c=openModal(L('关于这次研究'),'QWEN3-32B · L55_N11703','text-mode'),box=node('div','record-about');box.append(node('h3','',L('Colorado · Diverse 100M')));paragraph(box,L('记录日期为2026-10-29。本次运行配置的模型为deepseek-v4.1-flash-1017，推理级别为high。'));paragraph(box,L('逐步查看本次研究的任务、Skill、关键操作和交互报告。'));c.append(box);}
function initDemo(){if(location.hash&&steps.some(s=>s.id===location.hash.slice(1)))document.body.classList.remove('cover-open');$('#start-demo').onclick=()=>startDemo(0);$('#back-cover').onclick=showCover;$('#home-link').onclick=e=>{e.preventDefault();showCover();};$('#demo-prev').onclick=()=>select(Math.max(0,current-1));$('#demo-next').onclick=()=>select(current===steps.length-1?0:current+1);$('#about-demo').onclick=aboutDemo;$('#close-reading').onclick=()=>$('#reading-dialog').close();$('#reading-dialog').addEventListener('click',e=>{if(e.target!==e.currentTarget)return;const r=e.currentTarget.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.currentTarget.close();});$('#reading-dialog').addEventListener('close',()=>{if(!$('#reading-dialog').open)$('#reading-content').replaceChildren();});let coverId='published';document.querySelectorAll('[data-cover-snapshot]').forEach(b=>b.onclick=()=>{coverId=b.dataset.coverSnapshot;document.querySelectorAll('[data-cover-snapshot]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',x===b?'true':'false');});$('#cover-paper').src='screenshots/'+coverId+'.png';$('#cover-paper').alt=snapshot(coverId).title;$('#cover-paper-label').textContent=snapshot(coverId).claim.title;});$('#cover-zoom').onclick=()=>openImage('screenshots/'+coverId+'.png',snapshot(coverId).title,snapshot(coverId));document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>startDemo(steps.findIndex(s=>s.id===b.dataset.jump)));document.addEventListener('click',e=>{const a=e.target.closest('a');if(!a||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey||a.closest('.reading-actions'))return;const href=a.getAttribute('href');if(!/^content\/.+\.json$/.test(href||''))return;e.preventDefault();const c=openModal(a.textContent.replace(' ↗',''),L('JSON · 演示节选'),'text-mode');c.append(node('div','loading',L('正在读取记录……')));const revision=modalRevision;get(href).then(z=>{if(revision!==modalRevision||!$('#reading-dialog').open)return;c.replaceChildren();c.append(node('pre','inspector-text',text(z)));$('#reading-actions').append(link(L('单独打开JSON'),href));}).catch(x=>{c.textContent=x.message;});});}
function demoOverview(host,s){for(const [label,body]of s.story){const block=node('div','story-block');block.append(node('div','story-label',label),node('div','story-text',body));host.append(block);}if(s.id==='agent1'||s.id==='agent3')host.append(node('div','story-result',data.stages[s.output].claim.title));if(s.id==='publish')outputLink(host,L('完整交互报告'),snapshot('published').file);}
document.addEventListener('keydown',e=>{if($('#reading-dialog').open||document.body.classList.contains('cover-open')||e.target.getAttribute('role')==='tab')return;if(/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;if(e.key==='ArrowRight'){e.preventDefault();select(current+1);}if(e.key==='ArrowLeft'){e.preventDefault();select(current-1);}});
window.addEventListener('hashchange',()=>{const i=steps.findIndex(s=>s.id===location.hash.slice(1));if(i>=0&&i!==current)select(i);});
get('content/episode.json').then(d=>{data=d;$('#cover-paper-label').textContent=snapshot('published').claim.title;buildNav();initDemo();const i=steps.findIndex(s=>s.id===location.hash.slice(1));select(i>=0?i:0);}).catch(e=>{$('#detail').textContent=e.message;});
})();
