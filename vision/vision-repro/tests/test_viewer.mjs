import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';

const source=readFileSync(new URL('../src/vision_repro/web/microscope/app.js',import.meta.url),'utf8');
const element={addEventListener(){},classList:{add(){},remove(){}},querySelectorAll(){return []}};
const requests=[];
const context=vm.createContext({console,URLSearchParams,TextDecoder,Response,Blob,atob,Uint8Array,
  document:{querySelector(){return element},querySelectorAll(){return []},addEventListener(){}},
  window:{addEventListener(){}},location:{search:''},history:{replaceState(){}},
  fetch:async path=>{requests.push(path);return Response.json({exampleBuckets:[],heatBase64:'AECA/w=='});},
  setTimeout(){},requestAnimationFrame(fn){fn()}});
vm.runInContext(source.split("try{data=await json('data.json');")[0],context);
const run=code=>vm.runInContext(code,context);
run("data={sampleRoot:'samples/',defaultSteps:1024,publishedApi:false};state={layer:'conv2d0',channel:0,step:1024,variant:'positive'};ours={};renderDetail=()=>{};");
await run('loadPublished()');await run('loadExamplePublication()');
assert.deepEqual(requests,[],'static exports must not probe a publication API');
const html=run("sampleVisual({id:'47',image:'fixture',atlasUrl:'rf/images/example.webp',tile:47,atlasColumns:8,atlasRows:6,heatShape:[24,24],mapLocation:[3,3],activationPosition:[.25,.75],heatPeak:1,rank:1})");
assert.match(html,/background-size:800% 600%/);
assert.match(html,/background-position:100% 100%/);
assert.match(html,/left:75%;top:25%/);
run("data.rfExamples={nodes:['conv2d0:0'],nodeRoot:'rf/nodes/'}");
await run('loadSamples()');
assert.equal(requests.at(-1),'rf/nodes/conv2d0/0.json');
run('showHeat=true');await run('paintHeatmaps()');
assert.deepEqual(Array.from(await run("heatCache.get('conv2d0:0')")),[0,64,128,255]);
run("examplePublication={release:'rf_cloud',nodes:['conv2d0:0']};cache.clear();heatCache.clear()");
await run('loadSamples()');assert.equal(requests.at(-1),'/api/examples/rf_cloud/conv2d0/0');
run("state.channel=1;data.rfExamples={nodes:[],nodeRoot:'rf/nodes/'}");
await run('loadSamples()');assert.equal(requests.at(-1),'samples/neurons/conv2d0/1.json');
run("ours['conv2d0:0']={ours:{positive:{512:'old-512.png',1024:'old-1024.png'}},variantInfo:{positive:{config:{seed:7}}}};mergeOptimized({layer:'conv2d0',channel:0,ours:{},pending:{positive:'quality_review'}})");
assert.equal(run("ours['conv2d0:0'].ours.positive[512]"),'old-512.png');
assert.equal(run("ours['conv2d0:0'].ours.positive[1024]"),undefined);
assert.equal(run("ours['conv2d0:0'].legacyVariantInfo.positive.config.seed"),7);
assert.match(source,/loadSamples\(\);loadPublished\(\);\}\ncatch/);
console.log('PASS RF atlas placement, extremum coordinates, embedded heatmaps, static/cloud fallback and pending-result integrity.');
