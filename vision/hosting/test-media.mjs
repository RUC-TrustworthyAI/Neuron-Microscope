import assert from 'node:assert/strict';
import {createHash} from 'node:crypto';
import {mediaRoute} from './media.mjs';
const records=new Map();
const BUCKET={
 async put(key,raw,opts={}){records.set(key,{raw:Buffer.from(raw),...opts});},
 async head(key){const r=records.get(key);return r?{size:r.raw.length,customMetadata:r.customMetadata}:null;},
 async get(key){const r=records.get(key);return r?{size:r.raw.length,customMetadata:r.customMetadata,httpEtag:'"etag"',body:r.raw,json:async()=>JSON.parse(r.raw),writeHttpMetadata(h){h.set('Content-Type',r.httpMetadata?.contentType||'application/json');}}:null;}
};
const env={BUCKET,IMPORT_TOKEN:'test-only',IMPORT_EXPIRES_AT:new Date(Date.now()+60000).toISOString()};
const call=(path,method='GET',body,token=env.IMPORT_TOKEN)=>mediaRoute(new Request('https://example.test'+path,{method,headers:{'X-Microscope-Import':token},body}),env);
const hash=b=>createHash('sha256').update(b).digest('hex');
const put=async(key,raw)=>{const b=Buffer.from(typeof raw==='string'?raw:JSON.stringify(raw));return call('/api/import/object?key='+key+'&sha256='+hash(b),'PUT',b);};
assert.equal((await call('/api/import/object','PUT','x','wrong')).status,401);
assert.equal((await mediaRoute(new Request('https://example.test/api/import/verify'),{...env,IMPORT_EXPIRES_AT:'2000-01-01'})).status,401);
assert.equal((await call('/api/import/object?key=../bad&sha256='+'0'.repeat(64),'PUT','x')).status,400);
assert.equal((await call('/api/import/object?key=images/'+'0'.repeat(64)+'.webp&sha256='+'0'.repeat(64),'PUT','x')).status,400);
assert.equal((await call('/api/import/object?key=releases/test/catalog.json&sha256='+'0'.repeat(64),'PUT',Buffer.alloc(4*1024*1024+1))).status,400);
const release='test',catalog={release,snapshotAt:'2026-10-28',variants:1,images:1,neurons:1,nodes:{'conv2d0:0':'10000'}};
const manifest={release,variants:1,images:1,neurons:1};
assert.equal((await put('releases/test/catalog.json',catalog)).status,200);
assert.equal((await put('releases/test/catalog.json',catalog)).status,200);
assert.equal((await put('releases/test/catalog.json',{...catalog,variants:2})).status,409);
assert.equal((await put('releases/test/manifest.json',manifest)).status,200);
let r=await call('/api/import/verify','POST',JSON.stringify({objects:[{key:'releases/test/manifest.json',sha256:hash(JSON.stringify(manifest)),bytes:JSON.stringify(manifest).length}]}));assert.deepEqual((await r.json()).bad,[]);
r=await call('/api/import/publish','POST',JSON.stringify({release:'missing',manifestSha256:hash('x')}));assert.equal(r.status,409);
r=await call('/api/import/publish','POST',JSON.stringify({release,manifestSha256:hash(JSON.stringify(manifest))}));assert.equal(r.status,200);
assert.equal((await (await call('/api/published')).json()).release,'test');
assert.equal((await call('/api/catalog/test','HEAD')).body,null);
assert.equal((await mediaRoute(new Request('https://example.test/api/catalog/test',{headers:{'If-None-Match':'"etag"'}}),env)).status,304);
assert.equal((await call('/api/catalog/missing')).status,404);
assert.equal((await call('/api/catalog/test','POST','x')).status,405);
assert.equal(await call('/index.html'),null);
// RIFF's binary size word is not text and may contain valid multibyte UTF-8.
const webp=Buffer.concat([Buffer.from('RIFF'),Buffer.from([0xc2,0xa9,0,0]),Buffer.from('WEBPVP8L')]);
const digest=hash(webp);
const imageReply=await call('/api/import/object?key=images/'+digest+'.webp&sha256='+digest,'PUT',webp);
assert.equal(imageReply.status,200);
// Natural-example releases have their own atomic pointer; optimization is unchanged.
assert.equal((await (await call('/api/examples/published')).json()).release,null);
const examples={kind:'examples',release:'rf_test',neurons:1,samples:48,policy:'rf_fit'};
assert.equal((await put('examples/rf_test/manifest.json',examples)).status,200);
assert.equal((await put('examples/rf_test/catalog.json',{...examples,nodes:{'conv2d0:0':1}})).status,200);
assert.equal((await put('examples/rf_test/nodes/conv2d0/0.json',{channel:0})).status,200);
r=await call('/api/import/publish-examples','POST',JSON.stringify({release:'rf_test',manifestSha256:hash(JSON.stringify(examples))}));
assert.equal(r.status,200);
assert.equal((await (await call('/api/examples/published')).json()).release,'rf_test');
assert.deepEqual((await (await call('/api/examples/published')).json()).nodes,['conv2d0:0']);
assert.equal((await (await call('/api/published')).json()).release,'test');
assert.equal((await (await call('/api/examples/rf_test/conv2d0/0')).json()).channel,0);
assert.equal((await call('/api/examples/rf_test/conv2d0/0','HEAD')).body,null);
assert.equal((await call('/api/examples/missing/conv2d0/0')).status,404);
console.log('PASS R2 authentication, expiry, validation, limits, immutability, publication, reads.');
