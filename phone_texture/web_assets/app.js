'use strict';
const $ = id => document.getElementById(id);
const faces = ['front','back','left','right','top','bottom'];
const labels = {front:'正面',back:'背面',left:'左侧',right:'右侧',top:'顶面',bottom:'底面'};
const active = new Set(['starting','running','stopping','packaging']);
const statusLabels = {starting:'准备中',running:'处理中',stopping:'正在停止',packaging:'保存结果',complete:'已完成',failed:'处理失败',stopped:'内存不足',cancelled:'已停止',interrupted:'服务中断'};
let scanData = null, selected = new Set(), jobs = [], selectedJob = localStorage.getItem('sixview-job'), currentFace = 'back', galleryKey = '', toastTimer, busy = false, viewerData = null;
function node(tag, text, cls) { const el=document.createElement(tag); if(text!==undefined)el.textContent=text; if(cls)el.className=cls; return el; }
function notice(text, error=false) { $('toast').textContent=text; $('toast').className=error?'error':''; $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,error?10000:4500); }
async function api(path, body) {
 const r=await fetch('/api/'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-App-Token':document.querySelector('meta[name=app-token]').content},body:JSON.stringify(body)});
 const data=await r.json(); if(!r.ok)throw new Error(data.error||'请求未完成'); return data;
}
function asset(job,kind,sku,face,extra={}) { const params={job,kind,...extra}; if(sku)params.sku=sku;if(face)params.face=face;return '/api/artifact?'+new URLSearchParams(params); }
function seconds(n) { n=Math.round(n||0);return n<60?n+' 秒':Math.floor(n/60)+' 分 '+n%60+' 秒'; }
function updateConcurrency(){const value=$('concurrency').value;$('concurrencyValue').textContent=value;$('concurrency').setAttribute('aria-valuetext',value+' 并发');}
$('concurrency').oninput=updateConcurrency;updateConcurrency();
function updateStart(){const running=jobs.some(j=>active.has(j.status));$('shutdown').disabled=running;$('shutdown').title=running?'请先停止本批再退出':'停止本地网页服务';$('start').disabled=busy||running||!scanData||!selected.size||!$('model').value;$('start').textContent=running?'处理中…':selected.size?`开始处理 · ${selected.size} 台`:'开始处理';$('selectionCount').textContent=`已选 ${selected.size} 台`;$('selectAll').checked=!!scanData&&selected.size===scanData.valid&&selected.size>0;$('selectAll').indeterminate=selected.size>0&&selected.size<scanData.valid;}
function drawScan(){
 $('emptyInput').hidden=true;$('scanContent').hidden=false;$('scanCount').textContent=`${scanData.total} 台`;$('scanRows').replaceChildren();
 for(const row of scanData.rows){const tr=node('tr',undefined,row.valid?'':'invalid-row'),td=node('td'),label=node('label'),check=node('input');check.type='checkbox';check.checked=selected.has(row.name);check.disabled=!row.valid;check.setAttribute('aria-label','选择 '+row.name);check.onchange=()=>{check.checked?selected.add(row.name):selected.delete(row.name);updateStart();};label.append(check,document.createTextNode(row.name));td.append(label);const state=node('td');state.append(node('span',row.valid?'6 张':row.can_rename?'待命名':row.errors.join('；'),row.valid?'badge success':row.can_rename?'badge warn':'error-text'));tr.append(td,state);$('scanRows').append(tr);}
 const uncertain=scanData.total-scanData.valid-(scanData.renameable||0),skipped=scanData.skipped||[];
 $('scanNote').hidden=!uncertain&&!skipped.length;$('scanNote').textContent=skipped.length?`未识别，未处理：${skipped.join('、')}`:`${uncertain} 台待确认，暂不可处理`;
 $('namingBanner').hidden=!scanData.renameable;$('namingSummary').textContent=`${scanData.renameable} 台需要命名`;
 updateStart();
}
let scanVersion=0;
async function scanFolder(){
 clearTimeout(scanTimer);
 const path=$('source').value.trim();if(!path)return;
 const version=++scanVersion;busy=true;updateStart();$('scan').disabled=true;$('scan').textContent='…';$('scanCount').textContent='识别中…';
 try{const data=await api('scan',{path});if(version!==scanVersion)return;scanData=data;$('source').value=data.source;selected=new Set(data.rows.filter(r=>r.valid).map(r=>r.name));drawScan();}
 catch(e){if(version!==scanVersion)return;scanData=null;selected.clear();$('scanContent').hidden=true;$('emptyInput').hidden=false;$('scanCount').textContent='读取失败';notice(e.message,true);}
 finally{if(version===scanVersion){busy=false;$('scan').disabled=false;$('scan').textContent='↻';updateStart();}}
}
let namingScanId=null;
$('previewNames').onclick=()=>{
 namingScanId=scanData.id;$('namingPreview').replaceChildren();$('namingStatus').textContent='';
 for(const row of scanData.rows.filter(r=>r.can_rename)){
  const section=node('section',undefined,'naming-group');section.append(node('h3',row.name));const grid=node('div',undefined,'naming-grid');
  for(const face of faces){const item=row.recognition.images.find(i=>i.face===face),card=node('div',undefined,'naming-card'),img=node('img');img.loading='lazy';img.src='/api/scan-image?'+new URLSearchParams({scan:namingScanId,sku:row.name,name:item.name});img.alt=labels[face];card.append(img,node('span',item.name,'old-name'),node('strong','→ '+face+item.name.slice(item.name.lastIndexOf('.')).toLowerCase()));grid.append(card);}
  section.append(grid);$('namingPreview').append(section);
 }
 $('namingDialog').showModal();
};
$('closeNames').onclick=()=>$('namingDialog').close();
$('namingDialog').oncancel=e=>{if($('applyNames').disabled)e.preventDefault();};
$('applyNames').onclick=async()=>{
 const id=namingScanId,version=scanVersion;busy=true;updateStart();$('applyNames').disabled=true;$('closeNames').disabled=true;$('namingStatus').textContent='正在命名…';
 try{const data=await api('name',{scan_id:id});if(version===scanVersion){scanData=data;$('source').value=data.source;selected=new Set(data.rows.filter(r=>r.valid).map(r=>r.name));drawScan();}$('namingDialog').close();notice(`已命名 ${data.valid} 台`);}
 catch(e){$('namingStatus').textContent=e.message;}
 finally{busy=false;$('applyNames').disabled=false;$('closeNames').disabled=false;updateStart();}
};
$('scan').onclick=scanFolder;
let scanTimer;
$('source').oninput=()=>{clearTimeout(scanTimer);scanVersion++;busy=false;$('scan').disabled=false;$('scan').textContent='↻';scanData=null;selected.clear();$('scanContent').hidden=true;$('emptyInput').hidden=false;$('scanCount').textContent='';updateStart();scanTimer=setTimeout(()=>{if($('source').value.trim())scanFolder();},700);};
$('source').onkeydown=e=>{if(e.key==='Enter'){clearTimeout(scanTimer);scanFolder();}};
async function pick(kind){
 const target=$(kind==='input'?'source':'output');
 $('pickInput').disabled=true;$('pickOutput').disabled=true;
 try{const result=await api('pick-folder',{kind,path:target.value.trim()});if(result.cancelled)return;target.value=result.path;if(kind==='input')await scanFolder();}
 catch(e){notice(e.message,true);}
 finally{$('pickInput').disabled=false;$('pickOutput').disabled=false;}
}
$('pickInput').onclick=()=>pick('input');$('pickOutput').onclick=()=>pick('output');
$('selectAll').onchange=()=>{selected=new Set($('selectAll').checked?scanData.rows.filter(r=>r.valid).map(r=>r.name):[]);drawScan();};
async function startBatch(skus=Array.from(selected)){busy=true;updateStart();try{const j=await api('start',{scan_id:scanData.id,model:$('model').value,skus,concurrency:+$('concurrency').value,output:$('output').value.trim()});selectedJob=j.id;localStorage.setItem('sixview-job',j.id);galleryKey='';await refresh();$('jobPanel').scrollIntoView({behavior:'smooth',block:'start'});}catch(e){notice(e.message,true);}finally{busy=false;updateStart();}}
$('model').value=localStorage.getItem('sixview-model')||'';
$('model').onchange=()=>{localStorage.setItem('sixview-model',$('model').value);updateStart();};
$('start').onclick=()=>startBatch();
function stageText(r){if(r.status==='complete')return '完成 · '+seconds(r.seconds);if(r.status==='failed')return r.error||'处理失败，请查看日志';if(r.status==='waiting')return '排队中';if(r.status==='unfinished')return '未完成';return ({loading:'准备中',geometry:'抠图摆正',lighting:'优化光照',alignment:'结构对齐',checking:'检查保存'}[r.stage]||'正在处理')+(r.face?' · '+labels[r.face]:'');}
function currentJob(){return jobs.find(j=>j.id===selectedJob);}
function drawJob(){
 const j=currentJob();$('jobPanel').hidden=!j;if(!j){$('resultPanel').hidden=true;return;}
 $('jobTitle').textContent=active.has(j.status)?'处理中':'本批结果';if($('taskDetails').dataset.job!==j.id){$('taskDetails').open=active.has(j.status);$('taskDetails').dataset.job=j.id;}$('jobBadge').textContent=statusLabels[j.status]||j.status;$('jobBadge').className='badge push '+(j.status==='complete'?'success':['failed','stopped','interrupted'].includes(j.status)?'warn':'');
 $('jobId').textContent=new Date(j.created*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});$('jobId').title=j.id;$('jobConcurrency').textContent=`${j.total} 台 · ${j.concurrency} 并发${j.model==='iphone17promax'?' · Pro Max':j.model==='iphone17pro'?' · Pro':''}`;$('jobMessage').hidden=active.has(j.status)||j.status==='complete';$('jobMessage').textContent=j.message;$('doneCount').textContent=`${j.completed} / ${j.total}`;$('elapsed').textContent=seconds(j.elapsed);$('failedCount').textContent=j.failed;$('failedStat').hidden=!j.failed;$('progress').value=j.progress;$('progress').hidden=!active.has(j.status);
 $('taskRows').replaceChildren(...j.rows.map(r=>{const row=node('div',undefined,'task-row '+r.status);row.append(node('span',undefined,'state-dot'),node('span',r.name),node('span',stageText(r),'description'));return row;}));
 $('stop').hidden=!active.has(j.status);$('stop').disabled=['stopping','packaging'].includes(j.status);$('retry').hidden=active.has(j.status)||j.completed===j.total;$('retry').disabled=jobs.some(x=>active.has(x.status));$('log').href=asset(j.id,'log');
 $('resultPanel').hidden=!j.completed;
 if(j.completed){$('overview').hidden=!j.downloadable;$('download').hidden=!j.downloadable;$('overview').href=asset(j.id,'overview');$('download').href=asset(j.id,'zip',null,null,{download:1});drawGallery(j);}
}
function drawGallery(j){const key=j.id+'|'+currentFace+'|'+j.rows.filter(r=>r.preview).map(r=>r.name).join(',');if(key===galleryKey)return;galleryKey=key;$('gallery').replaceChildren();for(const r of j.rows.filter(x=>x.preview)){const card=node('button',undefined,'image-card'),wrap=node('div',undefined,'image-wrap checker'),img=node('img');img.loading='lazy';img.src=asset(j.id,'image',r.name,currentFace,{thumb:1});img.alt=r.name+' '+labels[currentFace]+'处理结果';wrap.append(img);const caption=node('div',undefined,'caption');caption.append(node('strong',r.name),node('span','↗'));card.append(wrap,caption);card.onclick=()=>showViewer(j,r.name,currentFace);$('gallery').append(card);}}
for(const f of faces){const b=node('button',labels[f],f===currentFace?'selected':'');b.onclick=()=>{currentFace=f;for(const c of $('faceTabs').children)c.classList.toggle('selected',c===b);const j=currentJob();if(j)drawGallery(j);};$('faceTabs').append(b);}
function showViewer(j,sku,face){viewerData={id:j.id,sku,face};$('viewerTitle').textContent=sku+' · '+labels[face];$('imageStage').classList.remove('actual');$('fit').classList.add('selected');$('actual').classList.remove('selected');viewerImage('image');$('downloadImage').href=asset(j.id,'image',sku,face,{download:1});$('viewer').showModal();}
function viewerImage(kind){const d=viewerData;if(!d)return;$('viewerImage').src=asset(d.id,kind,d.sku,d.face);$('showResult').classList.toggle('selected',kind==='image');$('showSource').classList.toggle('selected',kind==='source');}
$('showResult').onclick=()=>viewerImage('image');$('showSource').onclick=()=>viewerImage('source');$('closeViewer').onclick=()=>$('viewer').close();$('background').onclick=()=>$('imageStage').classList.toggle('dark');
$('fit').onclick=()=>{$('imageStage').classList.remove('actual');$('fit').classList.add('selected');$('actual').classList.remove('selected');};$('actual').onclick=()=>{$('imageStage').classList.add('actual');$('actual').classList.add('selected');$('fit').classList.remove('selected');};
$('stop').onclick=async()=>{const j=currentJob();if(!j)return;try{await api('cancel',{id:j.id});await refresh();}catch(e){notice(e.message,true);}};
$('openFolder').onclick=async()=>{try{await api('open',{id:selectedJob});}catch(e){notice(e.message,true);}};
$('retry').onclick=async()=>{const j=currentJob();if(!j)return;try{$('model').value=j.model||'';$('source').value=j.source;scanData=await api('scan',{path:j.source});selected=new Set(j.rows.filter(r=>r.status!=='complete').map(r=>r.name));$('output').value=j.root.slice(0,j.root.lastIndexOf('/'));$('concurrency').value=Math.min(3,j.concurrency);updateConcurrency();drawScan();$('intake').scrollIntoView({behavior:'smooth'});notice('已选中未完成项目，点击开始重试。');}catch(e){notice(e.message,true);}};
let historyKey='';
function drawHistory(){const key=jobs.map(j=>j.id+j.status+j.completed).join('|')+selectedJob;if(key===historyKey)return;historyKey=key;if(!jobs.length)return;$('history').replaceChildren(...jobs.slice(0,20).map(j=>{const b=node('button',undefined,'history-item'+(j.id===selectedJob?' selected':'')),info=node('span',undefined,'history-name');info.append(document.createTextNode(j.source.split('/').pop()+' · '+j.total+' 台'),node('small',new Date(j.created*1000).toLocaleString('zh-CN')+' · '+j.concurrency+' 并发'));b.append(info,node('span',statusLabels[j.status]||j.status,'badge'));b.onclick=()=>{selectedJob=j.id;localStorage.setItem('sixview-job',j.id);galleryKey='';drawJob();drawHistory();$('jobPanel').scrollIntoView({behavior:'smooth'});};return b;}));}
async function refresh(){try{const state=await api('state');jobs=state.jobs;if(!$('output').value)$('output').value=state.default_output;const m=state.host;$('available').textContent=(m.available/1024**3).toFixed(1)+' GB';$('memoryTotal').textContent=(m.total/1024**3).toFixed(0)+' GB';$('memoryBadge').textContent=m.pressure>=4?'压力较高':m.pressure===2?'稍有压力':'正常';$('memoryBadge').className='badge '+(m.pressure>=2?'warn':'success');$('connectionDot').classList.remove('offline');$('connectionText').textContent='本地运行';if(!jobs.find(j=>j.id===selectedJob))selectedJob=jobs[0]?.id;drawJob();drawHistory();updateStart();}catch(e){$('connectionDot').classList.add('offline');$('connectionText').textContent='未连接，请重新启动';$('start').disabled=true;}}
async function poll(){await refresh();setTimeout(poll,1500);}poll();

$('shutdown').onclick=async()=>{try{await api('shutdown',{});notice('服务已退出');}catch(e){notice(e.message,true);}};
