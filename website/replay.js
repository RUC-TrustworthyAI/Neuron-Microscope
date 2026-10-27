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
 {id:'initial',group:L('准备'),nav:L('初始化HTML'),role:L('固定程序'),title:L('先得到一份只有证据的页面'),snapshot:'initial',tab:'overview',
  summary:L('程序把已有激活统计和自然语料样本整理成HTML，供Agent 1读取。这份页面已经包含数据，解释正文和后续实验栏目等待补充。'),
  facts:[L('输入包含1B tokens的统计、自然上下文、原始gate／up／product，以及已有的输出方向分析。'),L('这次输入HTML已有434个散点；后续阶段会按各自证据集合重建选点。'),L('右侧展示的是实际输入文件的页面截图，可打开同一文件进行交互。')],output:'initial'},
 {id:'scan',group:L('准备'),nav:L('准备scan样本'),role:L('固定CPU程序'),title:L('读取已有1B激活，补充待检验样本'),snapshot:'initial',tab:'overview',
  summary:L('在Agent 1开始前，固定程序按激活幅度分层抽样。样本先保存在运行目录，随后用于检查初步解释的预测能力。'),
  facts:[L('14个幅度区间各抽取40个候选，共560个；再按候选池各区间占比分配并保留256个。'),L('256个点按文档分成128个feedback与128个final_holdout。'),L('这次扫描耗时约10.16秒，准备合计14.73秒，记录的新模型前向数为0。')],output:'scan'},
 {id:'agent1',group:L('Stage 1 · 初步解释'),nav:L('Agent 1读证据、写解释'),role:'Agent 1',title:L('根据真实激活，提出初步解释和规则'),trace:'agent1',snapshot:'agent1',tab:'prompt',
  summary:L('Agent 1先读取skill，再读取分布、正负样本、分位样本和输出效果。它最后提交claim及response_rules，交给固定程序继续处理。'),
  facts:[L('当次运行记录含32轮模型响应、38次工具调用；关键操作可在“工具轨迹”查看。'),L('当次提交了8条响应规则。后来加入的“最多3正、3负、1个No visible cue”要求，属于后续代码版本。'),L('初步解释的标题为“Colorado reference neuron”。页面写出发生在后面的固定打标完成之后。')],output:'01-interpretation'},
 {id:'feedback',group:L('Stage 1 · 初步解释'),nav:L('冻结解释，独立打标'),role:L('固定程序＋预测API'),title:L('让独立模型按固定解释预测，再接上真值'),snapshot:'agent1',aux:'initial-prediction',tab:'prompt',
  summary:L('初步解释提交后，程序固定其版本，把上下文与目标token交给独立预测器。预测器返回类别和方向后，程序才连接实测激活，形成反馈。'),
  facts:[L('预测器收到完整observed prefix和anchor，不收到实测activation、gate、up。'),L('这一轮处理178个已有样本与128个feedback点；保存结果共306条。'),L('该次记录为10批串行请求，固定打标约19分32秒。因此Stage 1总时长包含提交后的等待。')],output:'prediction-1'},
 {id:'agent2-attempt',group:L('Stage 2 · 实验证据'),nav:L('本轮Agent 2开始后中断'),role:'Agent 2',title:L('Agent 2读取解释和反馈，开始准备实验'),trace:'agent2-attempt',snapshot:'agent1',tab:'actions',
  summary:L('这一轮确实启动过Agent 2。它读取了初步解释、实测值和预测反馈，并开始定义语义组；后来运行切换为复用已有GPU测量，这次尝试在完成前中断。'),
  facts:[L('中断前保存了17轮模型响应、22次工具调用。'),L('它可以看到真实激活，并能通过response_predictions读取解释预测错误的案例。'),L('这条轨迹保留了中断前的实际行为，后面的实验素材由上一轮的完成记录提供。')],output:'attempt'},
 {id:'agent2-source',group:L('Stage 2 · 实验证据'),nav:L('展开被复用的实验来源'),branch:true,sub:L('较早一轮的Agent 2'),role:L('Agent 2 · 来源分支'),title:L('展开实验素材是怎样测出来的'),trace:'agent2-source',snapshot:'agent2',tab:'actions',
  summary:L('本轮沿用的测量来自pole-preserving-pair那次已完成的Agent 2。这里可以查看它当时怎样设计两轮实验、调用GPU工具、检查结果并整理Clusters。'),
  note:L('这一支发生在较早的运行中，使用当时的解释和skill。它提供后续导入的测量；本轮导入时保留了当前Agent 1的主解释。'),
  facts:[L('保存了51轮模型响应、50次工具调用；两轮具体实验计划可直接打开。'),L('它测量替换实验、3张RelP图、搜索、联合热图、消融和独立文本测试，再给自然成员整理语义Clusters。'),L('这些GPU测量的原始输入、位置和方法均随交接记录保留。')],output:'source'},
 {id:'agent2-import',group:L('Stage 2 · 实验证据'),nav:L('导入测量，补预测并交接'),role:L('固定程序'),title:L('把完成的测量接到当前解释后面'),snapshot:'agent2',aux:'experiment-prediction',tab:'overview',
  summary:L('切换到复用模式后，固定程序导入完成的实验与图注，沿用当前初步解释，再为新增实验输入补充盲预测，生成Stage 2交接HTML。'),
  facts:[L('主claim仍为“Colorado reference neuron”；程序没有在这个完成步骤改写它。'),L('页面新增3个定制图、7个语义Clusters、6条原始route、11条Family输入和Circuit。'),L('新增64个实验anchor；与已开放的306点合在一起，此时散点为370点。新增预测记录为3次API请求。')],output:'02-experiments'},
 {id:'agent3',group:L('Stage 3 · 修订解释'),nav:L('Agent 3读取结果并修订'),role:'Agent 3',title:L('结合反例和实验，正式改写解释'),trace:'agent3',snapshot:'agent3',tab:'prompt',
  summary:L('Agent 3接收初步解释、实验测量及打标反馈。它读取错误和边界案例，修订主解释、响应规则、Criticism、图注和单位标签，然后提交最终内容。'),
  facts:[L('保存了45轮模型响应、72次工具调用；实际任务要求使用已有测量，禁止新增实验。'),L('它把标题改为“Colorado local-information neuron”，并提交4条Criticism。'),L('这次仍提交了8条规则。修订稿及关键提交操作均可查看。')],output:'03-revision'},
 {id:'reader',group:L('Stage 3 · 修订解释'),nav:L('独立评阅最终稿'),role:L('固定程序＋评阅API'),title:L('再由一个独立模型评价解释的组织程度'),snapshot:'agent3',aux:'reader',tab:'prompt',
  summary:L('Agent 3提交时，程序调用独立paper reader。它根据claim、Criticism、强样本和实验摘要，给出An agent’s read的分类、评分及说明。'),
  facts:[L('评阅请求与研究agent的任务分开保存；可以查看模型收到的材料节选。'),L('当前实现给reader选择后的证据摘要，包含最多8个正向和8个负向强样本。'),L('评阅结果随后接入报告；这一调用不生成新GPU测量。')],output:'reader'},
 {id:'holdout',group:L('Stage 3 · 修订解释'),nav:L('冻结修订稿，再做盲预测'),role:L('固定程序＋预测API'),title:L('用固定的新解释做最后一次预测检查'),snapshot:'agent3',aux:'final-prediction',tab:'prompt',
  summary:L('修订稿提交后，程序固定新版本，应用到已有样本、实验输入和预留scan样本。预测完成后与实测值连接，再把结果写入图表。'),
  facts:[L('完成结果记录16批新预测；API归档中还保留了重试或后续尝试，相关操作在当次记录中保留。'),L('当次final_holdout分组记为127点，另1点因与复用测量重叠而标记为seen_reused。'),L('这127点中45点给出了正／负方向，44点正确，其余82点预测为不确定；相关分母在反馈中保留。')],output:'prediction-3'},
 {id:'render',group:L('Stage 3 · 修订解释'),nav:L('合并数据，写出完整HTML'),role:L('固定渲染程序'),title:L('把解释、测量和评阅结果组装成页面'),snapshot:'agent3',tab:'output',
  summary:L('固定代码把最终文字、数值、图表数据、打标结果和独立评阅合并进模板，写出Stage 3的交接HTML。交互来自模板中的JavaScript。'),
  facts:[L('到这里已有主解释、实验图、Clusters、Where it sits、Circuit、Tests、Criticism和整体评阅。'),L('散点恢复为498点，包括已有自然样本、scan和实验anchor。'),L('此时Spectrum仍沿用14个主题组；之后的刷新才把434个频谱点统一到冻结解释的预测规则。')],output:'render'},
 {id:'publish',group:L('发布与后续刷新'),nav:L('刷新打标，得到发布页面'),role:L('固定程序＋预测API'),title:L('补齐频谱打标，发布最终交互报告'),snapshot:'published',aux:'refresh',tab:'overview',
  summary:L('之后运行了频谱刷新：沿用修订稿，为256个自然scan点补预测，复用已完成结果，然后重新渲染发布副本，并安装双栏对照资源。'),
  facts:[L('刷新记录为8次API请求、新增256条预测、复用498条，新增模型前向数为0。'),L('频谱仍为434个自然点，组别变为8条响应规则加unresolved，共9组。'),L('右侧“单独打开完整HTML”指向本页保存的发布快照，保留图表、折叠和悬停交互。')],output:'publish'}
];
let data,current=0,activeTab='overview',version=0;
const tabNames={overview:L('过程讲解'),prompt:L('实际Prompt'),actions:L('工具轨迹'),output:L('阶段产物')};
const titles={headline:L('顶部主图'),custom:L('定制实验图'),graphs_curated:'Three graphs',spectrum:'Activation spectrum',clusters:L('语义Clusters'),routes:L('原始Routes'),anatomy:'Inside the neuron',family:'Where it sits',circuit:'Circuit',tests:L('Tests字段'),criticism:'Criticism',agent_read_full:L('独立整体评阅'),interpretation_costs:'Interpretation Costs'};
function snapshot(id){const saved=data.snapshots.find(s=>s.id===id);return {...saved,title:L(saved.title)};}
function traceMeta(id){return data.traces.find(t=>t.id===id);}
function auxFor(s){return data.aux_calls.filter(a=>a.kind===s.aux);}
function buildNav(){const nav=$('#step-nav');let last='';steps.forEach((s,i)=>{
 if(s.group!==last){nav.append(node('div','nav-group',s.group));last=s.group;}
 const b=node('button','step-button'+(s.branch?' branch':''));b.dataset.index=i;b.innerHTML='<span class="num">'+String(i+1).padStart(2,'0')+'</span><span>'+esc(s.nav)+(s.sub?'<small>'+esc(s.sub)+'</small>':'')+'</span>';b.onclick=()=>select(i);nav.append(b);
});}
function select(i,tab){current=Math.max(0,Math.min(steps.length-1,i));activeTab=tab||'overview';version++;const s=steps[current];if(!document.body.classList.contains('cover-open'))history.replaceState(null,'','#'+s.id);$('#step-counter').textContent=(current+1)+'／'+steps.length;document.querySelectorAll('.step-button').forEach((b,k)=>{b.classList.toggle('active',k===current);b.classList.toggle('past',k<current);b.setAttribute('aria-current',k===current?'step':'false');});
 const detail=$('#detail');detail.innerHTML='<div class="step-heading"><div class="step-eyebrow">'+esc(s.group)+'<span class="role">'+esc(s.role)+'</span></div><h2>'+esc(s.title)+'</h2><p>'+esc(s.summary)+'</p>'+(s.note?'<div class="branch-note">'+esc(s.note)+'</div>':'')+'</div><details class="step-facts"><summary>'+esc(L('查看本步的记录要点'))+'</summary><div class="facts"><ul>'+s.facts.map(f=>'<li>'+esc(f)+'</li>').join('')+'</ul></div></details><div class="tabs" role="tablist"></div><div class="tabbody"></div><div class="step-controls"></div>';
 const tabs=$('.tabs',detail);for(const [key,label] of Object.entries(tabNames)){const b=node('button',key===activeTab?'active':'',label);b.setAttribute('role','tab');b.setAttribute('aria-selected',key===activeTab?'true':'false');b.onclick=()=>changeTab(key);tabs.append(b);}
 const controls=$('.step-controls',detail),prev=node('button','',L('← 上一步')),next=node('button','primary',current===steps.length-1?L('已到最终页面'):L('下一步 →'));prev.disabled=!current;next.disabled=current===steps.length-1;prev.onclick=()=>select(current-1);next.onclick=()=>select(current+1);controls.append(prev,node('span','middle',String(current+1).padStart(2,'0')+'／'+steps.length),next);
 demoPreview(s);renderTab(s,version);syncDemoControls();updateLanguageLinks();if(!document.body.classList.contains('cover-open'))window.scrollTo({top:0,behavior:'auto'});
}
function changeTab(key){activeTab=key;version++;document.querySelectorAll('.tabs button').forEach((b,i)=>{const k=Object.keys(tabNames)[i];b.classList.toggle('active',k===key);b.setAttribute('aria-selected',k===key?'true':'false');});renderTab(steps[current],version);}
async function renderTab(s,v){const host=$('.tabbody');host.innerHTML='<div class="loading">'+esc(L('正在读取记录……'))+'</div>';try{
 if(activeTab==='overview'){host.innerHTML='';demoOverview(host,s);return;}
 if(activeTab==='prompt'){host.innerHTML='';if(s.trace){const t=await get('content/'+s.trace+'.json');if(v!==version)return;promptTrace(host,t);}else if(s.aux){await apiBrowser(host,s,v);}else{paragraph(host,L('这一段由固定程序执行，没有向LLM发出任务。下面保留它实际使用的参数或输入。'));code(host,L('执行记录'),s.id==='scan'?data.preparation:s.id==='agent2-import'?data.reuse_plan:snapshot(s.snapshot).claim);}
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
 const aux=t.id==='agent2-attempt'?'attempt-labeler':t.id==='agent2-source'?'blind-writer':null;
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
function fixedActions(host,s){if(s.id==='scan'){code(host,L('scan请求'),data.preparation['scan-request.json']);code(host,L('scan完成记录'),data.preparation['scan-summary.json']);code(host,L('准备与分割记录'),data.preparation['preparation.json']);return;}
 if(s.id==='agent2-import'){paragraph(host,L('按交接索引导入的测量如下。每项都来自已有文件；本步骤没有启动新的GPU实验。'));for(const m of data.measurements){const box=node('div','measurement');box.append(node('b','',m.id+' · '+m.tool),node('small','',m.title||m.status||''));host.append(box);}host.append(link(L('全部测量索引'),'content/measurements.json'));return;}
 if(s.id==='render'){paragraph(host,L('渲染顺序：读取最终claim及annotations → 合并保存的测量 → 接入独立评阅和盲预测 → 应用显示模板 → 写出HTML与交接文件。'));code(host,L('生成的报告'),{file:snapshot('agent3').file});host.append(link(L('查看对应HTML'),snapshot('agent3').file));return;}
 code(host,L('输入文件'),data.initial_files);}
async function apiBrowser(host,s,v){
 const calls=auxFor(s);paragraph(host,L('以下展示保存的Prompt与对应结果节选。涉及多样本任务时，保留完整的选中样本上下文。'));
 if(!calls.length){host.append(node('div','empty',L('本步骤没有Prompt示例。')));return;}
 for(const a of calls){const z=await get(a.file);if(v!==version)return;
  code(host,L('系统Prompt'),z.system,true);code(host,L('输入节选'),z.input_excerpt);code(host,L('对应结果节选'),z.output_excerpt);host.append(link(L('打开这个示例'),a.file));
 }
}
function stateLabel(v){if(!v?.present)return L('待补充');const list=[];if(v.count!=null)list.push(v.count+L('项'));for(const k of ['figures','points','groups','texts','family'])if(v[k]!=null)list.push(v[k]+({figures:L('个图'),points:L('个点'),groups:L('组'),texts:L('条输入'),family:L('行')}[k]));return list.join(language==='zh'?'，':', ')||L('已有字段／数据');}
function stateTable(host,before,after){paragraph(host,L('字段来自两份HTML中的实际数据。“已有字段”也可能包含占位值，具体内容可打开HTML检查。'));const table=node('table','field-list');table.innerHTML='<thead><tr><th>'+esc(L('版块'))+'</th><th>'+esc(before.title)+'</th><th>'+esc(after.title)+'</th></tr></thead>';const body=node('tbody');for(const [key,label] of Object.entries(titles)){const tr=node('tr');tr.append(node('td','',label),node('td','',stateLabel(before.fields[key])),node('td','',stateLabel(after.fields[key])));body.append(tr);}table.append(body);host.append(table);}
async function outputs(host,s,v){
 if(s.output==='initial'){code(host,L('初始化claim'),snapshot('initial').claim);paragraph(host,L('初始页中的原始统计和样本来自输入数据；解释及实验内容由后续步骤补充。'));return;}
 if(s.output==='scan'){code(host,L('scan交付数量'),data.preparation['preparation.json']);host.append(link(L('完整参数与扫描摘要'),'content/preparation.json'));return;}
 if(s.output==='prediction-1'||s.output==='prediction-3'){const st=s.output==='prediction-1'?'01-interpretation':'03-revision',p=data.stages[st].predictions[0];code(host,L('预测反馈'),p.result.summary);code(host,L('保存的具体反馈案例'),p.examples);host.append(link(L('反馈摘要与示例'),'content/episode.json'));return;}
 if(s.output==='source'||s.output==='attempt'){const t=await get('content/'+s.trace+'.json');if(v!==version)return;if(t.submission){code(host,L('保存的阶段提交'),await get(t.submission));host.append(link(L('阶段提交JSON'),t.submission));}else paragraph(host,L('这次尝试在完成前中断，没有阶段完成稿。关键操作节选可在工具轨迹中查看。'));for(const p of t.plans||[])outputLink(host,p.name,p.file);return;}
 if(s.output==='reader'){const a=auxFor(s)[0];if(a){const z=await get(a.file);if(v!==version)return;code(host,L('独立评阅结果节选'),z.output_excerpt);host.append(link(L('评阅示例JSON'),a.file));}return;}
 if(s.output==='render'){stateTable(host,snapshot('initial'),snapshot('agent3'));outputLink(host,L('Stage 3完整HTML'),snapshot('agent3').file);return;}
 if(s.output==='publish'){stateTable(host,snapshot('agent3'),snapshot('published'));outputLink(host,L('最终发布HTML'),snapshot('published').file);code(host,L('发布刷新记录'),data.refresh['status.json']);return;}
 const st=data.stages[s.output];if(st){
  const before=s.output==='03-revision'?data.stages['01-interpretation'].claim:null;
  if(before){const diff=node('div','diff');for(const [label,c] of [[L('Agent 1初稿'),before],[L('Agent 3修订稿'),st.claim]]){const box=node('div','diff-block');box.append(node('small','',label),node('div','paper-title',c.title),node('div','',c.lede));diff.append(box);}host.append(diff);}
  code(host,L('提交的claim'),st.claim);code(host,L('提交的response_rules'),st.response_rules);host.append(link(L('完整解释与Criticism'),st.output));
  if(s.output==='02-experiments')paragraph(host,L('这份主claim与Stage 1保持一致；本阶段补入测量、图表和注释。'));
 }
}
let modalRevision=0;
function openModal(title,kicker,mode){modalRevision++;const d=$('#reading-dialog');if(d.open)d.close();$('#reading-title').textContent=title;$('#reading-kicker').textContent=kicker;const c=$('#reading-content');c.replaceChildren();c.className='reading-content '+mode;$('#reading-actions').replaceChildren();d.showModal();return c;}
function openTextModal(title,all){const c=openModal(title,L('PROMPT · 运行记录'),'text-mode'),pre=node('pre','inspector-text',all);c.append(pre);const copy=node('button','',L('复制全文'));copy.onclick=async()=>{try{await navigator.clipboard.writeText(all);copy.textContent=L('已复制');}catch{copy.textContent=L('可选中文本复制');}};$('#reading-actions').append(copy);}
function openHTML(snap,section='head'){const c=openModal(snap.title,L('INTERACTIVE HTML · 保存的科学内容'),'iframe-mode');const loading=node('div','html-loading',L('正在载入保存的HTML……'));c.append(loading);const f=node('iframe');f.title=snap.title;f.src=snap.file+'#'+section;f.onload=()=>loading.remove();c.append(f);$('#reading-actions').append(link(L('单独打开'),snap.file+'#'+section));}
function openImage(src,title,snap,section='head'){const c=openModal(title,L('SAVED HTML · 实际页面截图'),'image-mode');const img=node('img');img.src=src;img.alt=title;c.append(img);if(snap){const b=node('button','',L('打开交互HTML'));b.onclick=()=>openHTML(snap,section);$('#reading-actions').append(b);}$('#reading-actions').append(link(L('查看图片原文件'),src));}
function previewItems(id){const h={id:'head',label:L('页面顶部'),file:'screenshots/'+id+'.jpg'};const list={initial:['distribution','exemplars'],agent1:['claim','anatomy'],agent2:['custom','clusters','family','circuit'],agent3:['claim','criticism','agents-read','anatomy'],published:['spectrum','anatomy','circuit']};const names={distribution:L('激活分布'),exemplars:L('自然样本'),claim:L('解释正文'),anatomy:'Inside the neuron',custom:L('实验图'),clusters:L('语义Clusters'),family:'Where it sits',circuit:'Circuit',criticism:'Criticism','agents-read':L('独立整体评阅'),spectrum:'Activation spectrum'};return [h,...(list[id]||[]).map(k=>({id:k,label:names[k],file:'screenshots/'+id+'-'+k+'.jpg'}))];}
function demoPreview(s){const snap=snapshot(s.snapshot),host=$('#preview'),items=previewItems(s.snapshot);const defaults={scan:'distribution',agent1:'claim',feedback:'anatomy','agent2-attempt':'anatomy','agent2-source':'custom','agent2-import':'clusters',agent3:'claim',reader:'agents-read',holdout:'anatomy',render:'criticism',publish:'spectrum'};let chosen=items.find(x=>x.id===defaults[s.id])||items[0];host.innerHTML='<div class="preview-card"><div class="preview-title"><span>'+esc(L('这时的HTML'))+'</span><span class="label">'+esc(snap.title)+'</span></div><div class="preview-tools"></div><button class="preview-picture" aria-label="'+esc(L('放大本步页面截图'))+'"><img class="preview-image" width="1164" height="655"></button><div class="preview-description"></div><div class="preview-actions"></div></div><p class="preview-note">'+(s.branch?L('本步回看测量的生成过程。截图展示本轮导入完成后的页面。'):L('截图可放大阅读；交互HTML保留了原页面的图表、折叠与悬停。'))+'</p>';
 const select=node('select');select.setAttribute('aria-label',L('选择预览版块'));items.forEach(x=>{const o=node('option','',x.label);o.value=x.id;select.append(o);});select.value=chosen.id;const zoom=node('button','',L('放大 ⤢'));$('.preview-tools',host).append(select,zoom);const img=$('img',host),open=node('button','',L('打开交互HTML')),standalone=link(L('单独打开'),snap.file);$('.preview-actions',host).append(open,standalone);const draw=()=>{img.src=chosen.file;img.alt=snap.title+' · '+chosen.label;img.onerror=()=>{if(chosen.id!=='head'){img.src='screenshots/'+s.snapshot+'.jpg';img.onerror=null;}};$('.preview-description',host).textContent=chosen.label+' · '+(chosen.id==='head'?L('对应阶段的页面顶部'):L('对应阶段的版块截图'));standalone.href=snap.file+'#'+chosen.id;};select.onchange=()=>{chosen=items.find(x=>x.id===select.value);draw();};zoom.onclick=()=>openImage(chosen.file,snap.title+' · '+chosen.label,snap,chosen.id);$('.preview-picture',host).onclick=zoom.onclick;open.onclick=()=>openHTML(snap,chosen.id);draw();}
function syncDemoControls(){if(!$('#demo-next'))return;const s=steps[current];$('#playback-caption').textContent=String(current+1).padStart(2,'0')+'／'+steps.length+' · '+s.group;const p=$('#playback-progress');p.replaceChildren();steps.forEach((x,i)=>{const b=node('button',i===current?'active':i<current?'past':'');b.setAttribute('aria-label',L('跳到第')+(i+1)+L('步：')+x.nav);b.onclick=()=>startDemo(i);p.append(b);});$('#demo-prev').disabled=current===0;$('#demo-next').innerHTML='<span>'+esc(current===steps.length-1?L('从头再看'):L('下一步：')+steps[current+1].nav)+'</span><span>→</span>';}
function startDemo(i=0){document.body.classList.remove('cover-open');select(i);window.scrollTo({top:0,behavior:'auto'});}
function showCover(){document.body.classList.add('cover-open');history.replaceState(null,'',location.pathname+location.search);updateLanguageLinks();window.scrollTo({top:0,behavior:'auto'});}
function aboutDemo(){const c=openModal(L('关于这段记录'),'RECORDED INVESTIGATION','text-mode'),box=node('div','record-about');box.append(node('h3','',L('一段真实的Colorado解释过程')));paragraph(box,L('案例是Qwen3-32B的L55/N11703。这轮运行于北京时间2026-10-27凌晨开始，研究模型使用DeepSeek high。演示从保存的记录中选取阶段指令、Prompt示例、关键操作和报告。'));paragraph(box,L('本轮Agent 2开始后中断，随后导入较早一轮的完成测量。回放把那次Agent 2的轨迹作为来源分支保留，并展示导入后的当前解释。'));paragraph(box,L('这份记录使用当时的8条响应规则。后来加入的“最多3正、3负、1个No visible cue”限制属于后续版本。'));paragraph(box,L('页面仅加载随演示保存的静态资源。交互报告保留科学内容与图表；完整API日志和运行环境文件留在本地。'));c.append(box);}
function initDemo(){if(location.hash&&steps.some(s=>s.id===location.hash.slice(1)))document.body.classList.remove('cover-open');$('#start-demo').onclick=()=>startDemo(0);$('#back-cover').onclick=showCover;$('#home-link').onclick=e=>{e.preventDefault();showCover();};$('#demo-prev').onclick=()=>select(Math.max(0,current-1));$('#demo-next').onclick=()=>select(current===steps.length-1?0:current+1);$('#about-demo').onclick=aboutDemo;$('#close-reading').onclick=()=>$('#reading-dialog').close();$('#reading-dialog').addEventListener('click',e=>{if(e.target!==e.currentTarget)return;const r=e.currentTarget.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)e.currentTarget.close();});$('#reading-dialog').addEventListener('close',()=>{if(!$('#reading-dialog').open)$('#reading-content').replaceChildren();});let coverId='published';document.querySelectorAll('[data-cover-snapshot]').forEach(b=>b.onclick=()=>{coverId=b.dataset.coverSnapshot;document.querySelectorAll('[data-cover-snapshot]').forEach(x=>{x.classList.toggle('active',x===b);x.setAttribute('aria-pressed',x===b?'true':'false');});$('#cover-paper').src='screenshots/'+coverId+'.jpg';$('#cover-paper').alt=snapshot(coverId).title;$('#cover-paper-label').textContent=snapshot(coverId).claim.title;});$('#cover-zoom').onclick=()=>openImage('screenshots/'+coverId+'.jpg',snapshot(coverId).title,snapshot(coverId));document.querySelectorAll('[data-jump]').forEach(b=>b.onclick=()=>startDemo(steps.findIndex(s=>s.id===b.dataset.jump)));document.addEventListener('click',e=>{const a=e.target.closest('a');if(!a||e.metaKey||e.ctrlKey||e.shiftKey||e.altKey||a.closest('.reading-actions'))return;const href=a.getAttribute('href');if(!/^content\/.+\.json$/.test(href||''))return;e.preventDefault();const c=openModal(a.textContent.replace(' ↗',''),L('JSON · 演示节选'),'text-mode');c.append(node('div','loading',L('正在读取记录……')));const revision=modalRevision;get(href).then(z=>{if(revision!==modalRevision||!$('#reading-dialog').open)return;c.replaceChildren();c.append(node('pre','inspector-text',text(z)));$('#reading-actions').append(link(L('单独打开JSON'),href));}).catch(x=>{c.textContent=x.message;});});}
function isKeyCall(c,i,total){const cmd=typeof c.arguments==='string'?c.arguments:(c.arguments?.command||'');return i===0||i===total-1||/SKILL\.md|run_experiments\.py|run_tests\.py|label_spectrum\.py|submit_interpretation\.py|submit_revision|submit_experiments|complete_stage|complete\.py|--section response_predictions|--section routes --ids/.test(cmd);}
function storyBlock(host,label,body){const d=node('div','story-block');d.append(node('div','story-label',label),node('div','story-text',body));host.append(d);return d;}
function storyResult(host,message){host.append(node('div','story-result',message));}
function demoOverview(host,s){const story={
 initial:[[L('先看到什么'),L('一张带有真实数据的HTML：激活分布、自然语料样本，以及每个位置的gate、up和product。')],[L('接下来要回答什么'),L('这个neuron在怎样的上下文中响应？先读取证据，再尝试用语言解释这些响应。')],[L('交给下一步'),L('保留输入HTML和结构化测量，让Agent 1可以回到具体样本核对。')]],
 scan:[[L('为什么补样本'),L('只盯着最强响应容易把解释写得过窄。程序从不同激活幅度区间补充样本，用于之后检验。')],[L('实际做了什么'),L('读取已有1B激活，先得到560个分层候选，再按候选池占比分配并保留256个点。')],[L('交给下一步'),L('128个feedback点用于提供错误反馈，另128个点预留给最终检查。')]],
 agent1:[[L('Agent 1读什么'),L('正负强样本、不同分位的上下文、分布与输出方向。具体证据可以通过样本ID继续读取。')],[L('它提出了什么'),L('初步解释把较强正响应联系到Colorado相关上下文，把负响应联系到部分非Colorado地区的记录与联系信息。')],[L('交给下一步'),L('提交claim与8条response_rules。这个版本随后被冻结，用于独立预测。')]],
 feedback:[[L('怎样检验一句解释'),L('独立预测器只看固定解释、上下文前缀和目标token，预测该位置的正向、负向或不确定响应。')],[L('怎样知道哪里错了'),L('预测结束后，程序连接实测product。预测与实测不一致的样本会保留，供后续实验和修订参考。')],[L('交给下一步'),L('本次形成306条已有样本与feedback预测。记录中的10批请求为串行执行。')]],
 'agent2-attempt':[[L('本轮发生了什么'),L('Agent 2先读取初步解释、真实激活和预测错误，准备实验；这次尝试在完成前被中断。')],[L('保留了哪些行为'),L('17轮模型响应、22次工具调用，以及它开始组织语义组和触发打标的记录。')],[L('接下来怎样继续'),L('随后切换为复用已有测量。下一步可以展开这些测量最初怎样生成。')]],
 'agent2-source':[[L('实验想区分什么'),L('保持一句话的框架与评分位置一致，只替换地名等线索，观察响应是否真的随假设中的因素变化。')],[L('测量来自哪里'),L('较早一轮的Agent 2设计了两轮实验，完成替换、搜索、归因、干预和独立文本测试。')],[L('交给当前运行'),L('原始输入、评分位置、数值和图注随测量导入；本轮继续沿用当前Agent 1的初步解释。')]],
 'agent2-import':[[L('接回了什么'),L('17份交接测量和相关注释，包括3个定制图、7个语义Clusters、Family、Circuit与Tests。')],[L('哪些输入需要再判断'),L('为新增的64个实验anchor补充当前解释的盲预测，连同已有证据交给修订阶段。')],[L('解释在这一步怎样处理'),L('主claim仍为Colorado reference neuron。正式改写留到Agent 3。')]],
 agent3:[[L('Agent 3读什么'),L('已有实验的测量、位置和对照，以及初步解释在哪些样本上预测错误。')],[L('解释怎样改变'),L('负极说法从较宽泛的区域关联，收窄到具体的非Colorado公共服务或记录边界；不吻合的案例继续保留。')],[L('交给下一步'),L('提交Colorado local-information neuron、修订规则、图注和4条Criticism。')]],
 reader:[[L('评阅者拿到什么'),L('最终claim、Criticism、选择后的正负强样本与实验摘要。')],[L('它评价什么'),L('解释的组织程度、已得到支持的部分和仍未解释清楚的部分。分类与评分属于评阅模型的判断。')],[L('交给下一步'),L('结果进入An agent’s read，并与实验和解释一起保留在报告中。')]],
 holdout:[[L('怎样检查修订稿'),L('再次冻结解释，给独立预测器提供上下文与anchor，然后连接实测值检查方向。')],[L('这份记录给出了什么'),L('127个final_holdout点中，45个得到了正负方向判断，44个方向吻合；其余82个被判为不确定。')],[L('哪些信息继续保留'),L('预测、真值、样本分组和错误案例一并存档；与复用测量重叠的点单独标记。')]],
 render:[[L('把哪些东西组装起来'),L('解释文字、实验数值、归因图、Clusters、独立预测、Criticism与整体评阅。')],[L('固定代码做什么'),L('把这些内容合并到HTML模板，接上图表、悬停、折叠和费用记录。')],[L('此时仍有什么后续'),L('Stage 3交接页的Spectrum仍沿用14个主题组。之后的一次刷新把它接到冻结规则的预测结果。')]],
 publish:[[L('最后补齐了什么'),L('为256个自然scan点补充预测，复用已有结果，重新渲染发布副本。')],[L('得到怎样的报告'),L('434个频谱点对应8条规则加unresolved，合计9组；498个散点和已有GPU测量继续保留。')],[L('可以继续怎样阅读'),L('打开完整HTML，检查解释、实验、反例和剩余问题；也可以回到任何一步追溯它们的来源。')]]
 };(story[s.id]||[]).forEach(([label,body])=>storyBlock(host,label,body));
 if(s.id==='scan')storyResult(host,L('实际记录：扫描约10.16秒，准备共14.73秒，新模型前向数为0。'));
 if(s.id==='agent1')storyResult(host,L('初步标题：')+data.stages['01-interpretation'].claim.title);
 if(s.id==='feedback'){const records=data.stages['01-interpretation'].predictions[0].examples;const r=records.find(x=>x.predicted_pole!=='uncertain'&&x.correct===false);if(r)exampleDisagreement(host,r);}
 if(s.id==='agent2-source')renderMeasuredComparison(host);
 if(s.id==='agent3')storyResult(host,L('修订后的标题：')+data.stages['03-revision'].claim.title);
 if(s.id==='publish'){const a=link(L('打开完整报告'),snapshot('published').file);a.className='start-button';a.style.cssText='display:inline-flex;margin-top:18px;color:white';host.append(a);}
}
function exampleDisagreement(host,r){const box=node('div','small-evidence');box.append(node('div','story-label',L('一个实际的预测分歧')));const values=node('div','evidence-values');values.append(node('span','',L('预测：')+r.predicted_pole),node('span','',L('实测product：')+r.activation));box.append(values);const prefix=r.observed_prefix||'';box.append(node('div','story-label',L('上下文节选')),node('div','quote','……'+prefix.slice(-230)));const b=node('button','quiet-button',L('阅读完整上下文与预测'));b.onclick=()=>openTextModal(L('预测分歧的实际记录'),text(r));box.append(b);host.append(box);}
async function renderMeasuredComparison(host){const v=version;let z;try{z=await get('content/demo-evidence.json');}catch{return;}if(v!==version)return;const c=z.headline?.comparison;if(!c?.fillers)return;const box=node('div','small-evidence');box.append(node('div','story-label',L('一个实际的匹配比较')));box.append(node('div','science-template',c.template));paragraph(box,L('这些条件都测同一个句末“ at”。点击一行，查看实际输入与测量位置。'),'science-note');const rows=node('div','science-bars'),detail=node('div','science-note');box.append(rows,detail);const values=c.fillers.map(x=>x.activation),max=Math.max(...values.map(Math.abs));c.fillers.forEach(r=>{const b=node('button','science-row');const track=node('span','science-track'),zero=node('i','zero'),bar=node('i','bar');const magnitude=Math.abs(r.activation)/max*80;bar.style.left=(r.activation<0?15-magnitude:15)+'%';bar.style.width=magnitude+'%';bar.style.background=r.activation<0?'#5786a6':'#cb7950';track.append(zero,bar);b.append(node('span','',r.tag),track,node('span','val',Number(r.activation.toPrecision(3)).toString()));b.onclick=()=>{rows.querySelectorAll('button').forEach(x=>x.classList.toggle('active',x===b));detail.textContent=r.full_text+L('　｜　目标token：')+JSON.stringify(r.peak_token)+L('，位置：')+r.peak_index;};rows.append(b);});host.append(box);}

document.addEventListener('keydown',e=>{if($('#reading-dialog').open||document.body.classList.contains('cover-open')||e.target.getAttribute('role')==='tab')return;if(/INPUT|TEXTAREA|SELECT/.test(e.target.tagName)||e.ctrlKey||e.metaKey||e.altKey)return;if(e.key==='ArrowRight'){e.preventDefault();select(current+1);}if(e.key==='ArrowLeft'){e.preventDefault();select(current-1);}});
window.addEventListener('hashchange',()=>{const i=steps.findIndex(s=>s.id===location.hash.slice(1));if(i>=0&&i!==current)select(i);});
get('content/episode.json').then(d=>{data=d;buildNav();initDemo();const i=steps.findIndex(s=>s.id===location.hash.slice(1));select(i>=0?i:0);}).catch(e=>{$('#detail').textContent=e.message;});
})();
