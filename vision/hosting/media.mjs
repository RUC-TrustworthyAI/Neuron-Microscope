const shaPattern=/^[a-f0-9]{64}$/;
const releasePattern=/^[a-zA-Z0-9_-]{1,80}$/;
const keyPattern=/^(?:images\/[a-f0-9]{64}\.webp|(?:releases|examples)\/[a-zA-Z0-9_-]{1,80}\/(?:catalog\.json|manifest\.json|nodes\/(?:conv2d[012]|mixed[345][a-e])\/[0-9]{1,4}\.json))$/;
const MAX_BYTES=4*1024*1024;
const json=(value,status=200)=>Response.json(value,{status,headers:{'Cache-Control':'no-store'}});
const hex=b=>Array.from(new Uint8Array(b),x=>x.toString(16).padStart(2,'0')).join('');
async function bytes(request){
  const reader=request.body?.getReader();if(!reader)return new Uint8Array();
  const chunks=[];let length=0;
  for(;;){const {done,value}=await reader.read();if(done)break;length+=value.length;if(length>MAX_BYTES){await reader.cancel();throw new Error('too_large');}chunks.push(value);}
  const result=new Uint8Array(length);let pos=0;for(const chunk of chunks){result.set(chunk,pos);pos+=chunk.length;}return result;
}
async function authenticated(request,env){
  const expires=Date.parse(env.IMPORT_EXPIRES_AT||'');
  if(!env.IMPORT_TOKEN||!Number.isFinite(expires)||Date.now()>=expires)return false;
  const supplied=request.headers.get('X-Microscope-Import')||'';
  if(supplied.length!==env.IMPORT_TOKEN.length)return false;
  let diff=0;for(let i=0;i<supplied.length;i++)diff|=supplied.charCodeAt(i)^env.IMPORT_TOKEN.charCodeAt(i);
  return diff===0;
}
async function serve(request,bucket,key,cache){
  const object=await bucket.get(key);if(!object)return json({error:'not_found'},404);
  const headers=new Headers();object.writeHttpMetadata(headers);headers.set('ETag',object.httpEtag);
  headers.set('Cache-Control',cache);headers.set('X-Content-Type-Options','nosniff');
  if(request.headers.get('If-None-Match')===object.httpEtag)return new Response(null,{status:304,headers});
  headers.set('Content-Length',String(object.size));
  return new Response(request.method==='HEAD'?null:object.body,{headers});
}
export async function mediaRoute(request,env){
  const url=new URL(request.url),pathname=url.pathname;
  if(!pathname.startsWith('/api/')&&!pathname.startsWith('/media/'))return null;
  if(!env.BUCKET)return json({error:'storage_unavailable'},503);
  if(pathname.startsWith('/api/import/')){
    if(!await authenticated(request,env))return json({error:'unauthorized'},401);
    try{
      if(pathname==='/api/import/batch'&&request.method==='POST'){
        const {objects}=JSON.parse(new TextDecoder().decode(await bytes(request)));
        if(!Array.isArray(objects)||!objects.length||objects.length>16||objects.some(x=>!keyPattern.test(x.key||'')||!shaPattern.test(x.sha256||'')||typeof x.base64!=='string'))return json({error:'invalid_batch'},400);
        const results=[];
        for(let i=0;i<objects.length;i+=4){
          results.push(...await Promise.all(objects.slice(i,i+4).map(async x=>{
            const raw=Uint8Array.from(atob(x.base64),c=>c.charCodeAt(0));
            const nested=new Request(`${url.origin}/api/import/object?key=${encodeURIComponent(x.key)}&sha256=${x.sha256}`,{method:'PUT',headers:request.headers,body:raw});
            const r=await mediaRoute(nested,env);return {status:r.status,...await r.json()};
          })));
        }
        return json({results},results.every(x=>x.status===200)?200:409);
      }
      if(pathname==='/api/import/object'&&request.method==='PUT'){
        const key=url.searchParams.get('key'),sha=url.searchParams.get('sha256');
        if(!keyPattern.test(key||'')||!shaPattern.test(sha||''))return json({error:'invalid_key'},400);
        const raw=await bytes(request);
        if(!raw.length)return json({error:'empty_object'},400);
        const digest=hex(await crypto.subtle.digest('SHA-256',raw));
        if(digest!==sha)return json({error:'checksum_mismatch'},400);
        if(key.startsWith('images/')){
          const decoder=new TextDecoder();
          if(decoder.decode(raw.slice(0,4))!=='RIFF'||decoder.decode(raw.slice(8,12))!=='WEBP'||!key.includes(digest))return json({error:'invalid_image'},400);
        }else JSON.parse(new TextDecoder().decode(raw));
        const previous=await env.BUCKET.head(key);
        if(previous){
          if(previous.customMetadata?.sha256!==sha||previous.size!==raw.length)return json({error:'immutable_conflict'},409);
          return json({key,sha256:sha,bytes:raw.length,existing:true});
        }
        await env.BUCKET.put(key,raw,{httpMetadata:{contentType:key.endsWith('.webp')?'image/webp':'application/json; charset=utf-8'},customMetadata:{sha256:sha}});
        return json({key,sha256:sha,bytes:raw.length});
      }
      if(pathname==='/api/import/verify'&&request.method==='POST'){
        const {objects}=JSON.parse(new TextDecoder().decode(await bytes(request)));
        if(!Array.isArray(objects)||objects.length>100||objects.some(x=>!keyPattern.test(x.key||'')||!shaPattern.test(x.sha256||'')||!Number.isSafeInteger(x.bytes)||x.bytes<=0))return json({error:'invalid_manifest'},400);
        const bad=[];
        for(let i=0;i<objects.length;i+=8)await Promise.all(objects.slice(i,i+8).map(async x=>{
          const o=await env.BUCKET.head(x.key);if(!o||o.size!==x.bytes||o.customMetadata?.sha256!==x.sha256)bad.push(x.key);
        }));
        return json({checked:objects.length,bad});
      }
      if(pathname==='/api/import/publish'&&request.method==='POST'){
        const body=JSON.parse(new TextDecoder().decode(await bytes(request)));
        if(!releasePattern.test(body.release||'')||!shaPattern.test(body.manifestSha256||''))return json({error:'invalid_release'},400);
        const manifest=await env.BUCKET.get(`releases/${body.release}/manifest.json`);
        const catalog=await env.BUCKET.get(`releases/${body.release}/catalog.json`);
        if(!manifest||!catalog||manifest.customMetadata?.sha256!==body.manifestSha256)return json({error:'release_missing'},409);
        const m=await manifest.json(),c=await catalog.json();
        if(m.release!==body.release||c.release!==body.release||m.variants!==c.variants||m.neurons!==Object.keys(c.nodes).length)return json({error:'release_mismatch'},409);
        const active={release:body.release,publishedAt:new Date().toISOString(),snapshotAt:c.snapshotAt,variants:c.variants,images:c.images,neurons:m.neurons};
        await env.BUCKET.put('active.json',JSON.stringify(active),{httpMetadata:{contentType:'application/json; charset=utf-8'}});
        return json(active);
      }
      if(pathname==='/api/import/publish-examples'&&request.method==='POST'){
        const body=JSON.parse(new TextDecoder().decode(await bytes(request)));
        if(!releasePattern.test(body.release||'')||!shaPattern.test(body.manifestSha256||''))return json({error:'invalid_release'},400);
        const manifest=await env.BUCKET.get(`examples/${body.release}/manifest.json`);
        const catalog=await env.BUCKET.get(`examples/${body.release}/catalog.json`);
        if(!manifest||!catalog||manifest.customMetadata?.sha256!==body.manifestSha256)return json({error:'release_missing'},409);
        const m=await manifest.json(),c=await catalog.json();
        if(m.kind!=='examples'||m.release!==body.release||c.release!==body.release||m.samples!==c.samples||m.neurons!==Object.keys(c.nodes||{}).length||m.samples!==m.neurons*48)return json({error:'release_mismatch'},409);
        const active={release:body.release,publishedAt:new Date().toISOString(),snapshotAt:c.snapshotAt,neurons:m.neurons,samples:m.samples,policy:m.policy,nodes:Object.keys(c.nodes)};
        await env.BUCKET.put('examples-active.json',JSON.stringify(active),{httpMetadata:{contentType:'application/json; charset=utf-8'}});
        return json(active);
      }
      return json({error:'method_not_allowed'},405);
    }catch(error){return json({error:error.message==='too_large'?'too_large':'invalid_request'},400);}
  }
  if(!['GET','HEAD'].includes(request.method))return json({error:'method_not_allowed'},405);
  if(pathname==='/api/published'){
    const active=await env.BUCKET.get('active.json');return active?new Response(request.method==='HEAD'?null:active.body,{headers:{'Content-Type':'application/json','Cache-Control':'no-store'}}):json({release:null});
  }
  if(pathname==='/api/examples/published'){
    const active=await env.BUCKET.get('examples-active.json');return active?new Response(request.method==='HEAD'?null:active.body,{headers:{'Content-Type':'application/json','Cache-Control':'no-store'}}):json({release:null});
  }
  let match;
  if((match=pathname.match(/^\/media\/([a-f0-9]{64})\.webp$/)))return serve(request,env.BUCKET,`images/${match[1]}.webp`,'public, max-age=31536000, immutable');
  if((match=pathname.match(/^\/api\/catalog\/([a-zA-Z0-9_-]{1,80})$/)))return serve(request,env.BUCKET,`releases/${match[1]}/catalog.json`,'public, max-age=31536000, immutable');
  if((match=pathname.match(/^\/api\/neuron\/([a-zA-Z0-9_-]{1,80})\/(conv2d[012]|mixed[345][a-e])\/([0-9]{1,4})$/)))return serve(request,env.BUCKET,`releases/${match[1]}/nodes/${match[2]}/${match[3]}.json`,'public, max-age=31536000, immutable');
  if((match=pathname.match(/^\/api\/examples\/([a-zA-Z0-9_-]{1,80})\/(conv2d[012]|mixed[345][a-e])\/([0-9]{1,4})$/)))return serve(request,env.BUCKET,`examples/${match[1]}/nodes/${match[2]}/${match[3]}.json`,'public, max-age=31536000, immutable');
  return json({error:'not_found'},404);
}
