// Exercise the real viewer with document-replacing report loaders and scroll events.
// This models iframe lifecycle and scrolling without a browser dependency.
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';

const viewer = fs.readFileSync(new URL('../tools/comparison_assets/viewer.js', import.meta.url), 'utf8');

class Events {
  listeners = new Map();
  addEventListener(type, fn, options = {}) {
    const list = this.listeners.get(type) || [];
    list.push({fn, once:options.once}); this.listeners.set(type, list);
  }
  emit(type, data = {}) {
    const entries = [...(this.listeners.get(type) || [])];
    for (const item of entries) {
      if (item.once) this.listeners.set(type, this.listeners.get(type).filter(x => x !== item));
      item.fn({type, preventDefault() {}, ...data});
    }
  }
}
class Element extends Events {
  constructor(doc, tag, y = 0) {
    super(); this.ownerDocument=doc; this.tagName=tag; this.y=y;
    this.dataset={}; this.attributes={}; this.children=[]; this.offsetHeight=300;
    this.style={setProperty(key,value){this[key]=value;}};
    const classes=new Set();
    this.classList={add:x=>classes.add(x),remove:x=>classes.delete(x),contains:x=>classes.has(x)};
  }
  setAttribute(key,value) {
    this.attributes[key]=value;
    if(key==='id'){this.id=value;this.ownerDocument.ids.set(value,this);}
    if(key==='hidden')this.hidden=true;
  }
  getAttribute(key){return this.attributes[key];}
  append(...nodes){this.children.push(...nodes);}
  prepend(node){this.children.unshift(node);}
  before(){}
  after(){}
  focus(){}
  querySelector(selector){return selector==='.np-compare-arrow'?new Element(this.ownerDocument,'path'):null;}
  querySelectorAll(){return [];}
  getBoundingClientRect(){return {top:this.y-this.ownerDocument.defaultView.scrollY,height:50};}
  compareDocumentPosition(other){return this.y<=other.y?4:2;}
}
class Frame extends Element {
  constructor(doc){super(doc,'iframe');this.replaceDocument(false);}
  replaceDocument(complete){
    this.contentWindow=new ReportWindow();
    this.contentDocument=new Document(this.contentWindow,complete);
  }
}
class Document extends Events {
  constructor(win,complete=false){
    super();this.defaultView=win;win.document=this;this.ids=new Map();this.readyState='complete';
    this.documentElement=new Element(this,'html');this.body=new Element(this,'body');this.head=new Element(this,'head');
    if(complete){
      for(const [id,y] of [['nav',0],['head',200],['main',500],['claim',500],['exemplars',1500]]){
        const node=new Element(this,'section',y);node.setAttribute('id',id);
      }
    }
  }
  createElement(tag){return tag==='iframe'?new Frame(this):new Element(this,tag,10000);}
  getElementById(id){return this.ids.get(id)||null;}
  querySelector(){return null;}
  querySelectorAll(selector){
    return selector==='main > section[id], main > .prf[id]'?['claim','exemplars'].map(id=>this.getElementById(id)).filter(Boolean):[];
  }
}
class ReportWindow extends Events {
  scrollY=0;
  scrollTo({top}){if(this.scrollY!==top){this.scrollY=top;this.emit('scroll');}}
}
function setup({search='',loading=false,embedded=false,reference=true,path='/L50_N3507/'}={}){
  const win=new ReportWindow(), doc=new Document(win,true);
  win.top=embedded?{}:win;win.self=win;win.innerWidth=1400;
  doc.readyState=loading?'loading':'complete';
  doc.currentScript={dataset:{neuron:'L50_N3507',reference:reference?'comparison/original.html':'',original:'https://already-clean.goodfire.com/neuron-scientist/L50_N3507/'}};
  let next=0;const raf=new Map();
  const context={window:win,document:doc,location:{href:'https://reports.example'+path+search,search},URL,URLSearchParams,
    requestAnimationFrame:fn=>{const id=++next;raf.set(id,fn);return id;},cancelAnimationFrame:id=>raf.delete(id),
    setTimeout:()=>0,clearTimeout(){},ResizeObserver:class{observe(){}}};
  vm.runInNewContext(viewer,context);
  return {win,doc,frames:()=>['left','right'].map(side=>doc.getElementById('np-compare-'+side+'-pane').children.find(x=>x.tagName==='iframe')),
    flush(){let steps=0;while(raf.size){assert.ok(++steps<20,'scroll events must settle');const callbacks=[...raf.values()];raf.clear();callbacks.forEach(fn=>fn());}}};
}
function loadReports(app){for(const frame of app.frames()){frame.replaceDocument(true);frame.emit('load');}}
function checkBothDirections(app){
  const [left,right]=app.frames().map(f=>f.contentWindow);
  left.scrollTo({top:850});app.flush();assert.equal(right.scrollY,850);
  right.scrollTo({top:1700});app.flush();assert.equal(left.scrollY,1700);
  right.scrollTo({top:200});app.flush();assert.equal(left.scrollY,200);
}

test('direct links open comparison by default; both panes synchronize',()=>{
  const app=setup();assert.equal(app.doc.getElementById('np-compare-shell').hidden,false);
  assert.equal(app.doc.getElementById('np-compare-back').getAttribute('href'),'https://reports.example/');
  loadReports(app);assert.equal(app.doc.getElementById('np-compare-shell').dataset.ready,'true');checkBothDirections(app);
});
test('gzip loaders must not consume iframe readiness or scroll listeners',()=>{
  const app=setup({search:'?compare=0'});
  app.doc.getElementById('np-compare-toggle').emit('click');
  const [left,right]=app.frames();
  left.emit('load');right.emit('load');
  assert.equal(app.doc.getElementById('np-compare-shell').dataset.ready,undefined);
  left.replaceDocument(true);left.emit('load');
  assert.equal(app.doc.getElementById('np-compare-shell').dataset.ready,undefined);
  right.replaceDocument(true);right.emit('load');checkBothDirections(app);
  // A duplicate load must not reconnect spacers and listeners.
  const count=left.contentWindow.listeners.get('scroll').length;left.emit('load');
  assert.equal(left.contentWindow.listeners.get('scroll').length,count);
});
test('comparison waits for the outer document and preserves an opt-out',()=>{
  const app=setup({loading:true});assert.equal(app.doc.getElementById('np-compare-shell').hidden,true);
  app.win.emit('load');assert.equal(app.doc.getElementById('np-compare-shell').hidden,false);
  const single=setup({search:'?compare=0'});assert.equal(single.doc.getElementById('np-compare-shell').hidden,true);
  assert.equal(single.doc.getElementById('np-report-back').getAttribute('href'),'https://reports.example/');
  single.doc.getElementById('np-compare-toggle').emit('click');loadReports(single);checkBothDirections(single);
  const independent=setup({reference:false,path:'/experiments/run/100m/L50_N3507/index.html',search:'?compare=0'});
  assert.equal(independent.doc.getElementById('np-compare-shell'),null);
  assert.equal(independent.doc.getElementById('np-report-back').getAttribute('href'),'https://reports.example/experiments/run/100m/');
});
test('new input from the opposite pane wins over a queued scroll; close and reopen work',()=>{
  const app=setup();loadReports(app);const [left,right]=app.frames().map(f=>f.contentWindow);
  left.scrollTo({top:500});right.scrollTo({top:900});app.flush();assert.equal(left.scrollY,900);assert.equal(right.scrollY,900);
  app.doc.getElementById('np-compare-close').emit('click');assert.equal(app.doc.getElementById('np-compare-shell').hidden,true);
  app.doc.getElementById('np-compare-toggle').emit('click');assert.equal(app.doc.getElementById('np-compare-shell').hidden,false);checkBothDirections(app);
});
test('an embedded report does not open a recursive comparison',()=>{
  const app=setup({embedded:true});assert.equal(app.doc.getElementById('np-compare-shell'),null);
  assert.equal(app.doc.getElementById('np-report-back'),null);
});
