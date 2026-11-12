'use strict';
const $ = id => document.getElementById(id);
const faces = ['front','back','left','right','top','bottom'];
const labels = {front:'正面',back:'背面',left:'左侧',right:'右侧',top:'顶面',bottom:'底面'};
const active = new Set(['starting','running','stopping','packaging']);
const statusLabels = {starting:'准备中',running:'处理中',stopping:'正在停止',packaging:'保存结果',complete:'已生成',partial:'部分已生成',review:'已生成 · 待核对',failed:'处理失败',stopped:'内存不足',cancelled:'已停止',interrupted:'服务中断'};
let scanData = null, selected = new Set(), jobs = [], selectedJob = localStorage.getItem('sixview-job'), currentFace = 'back', galleryKey = '', toastTimer, busy = false, viewerData = null;
function selectedModel(){return document.querySelector('input[name="model"]:checked')?.value||'';}
function setModel(value){document.querySelectorAll('input[name="model"]').forEach(input=>{input.checked=input.value===value;});}
function node(tag, text, cls) { const el=document.createElement(tag); if(text!==undefined)el.textContent=text; if(cls)el.className=cls; return el; }
function notice(text, error=false) { $('toast').textContent=text; $('toast').className=error?'error':''; $('toast').hidden=false; clearTimeout(toastTimer); toastTimer=setTimeout(()=>$('toast').hidden=true,error?10000:4500); }
async function api(path, body) {
 const r=await fetch('/api/'+path,body===undefined?{}:{method:'POST',headers:{'Content-Type':'application/json','X-App-Token':document.querySelector('meta[name=app-token]').content},body:JSON.stringify(body)});
 const data=await r.json(); if(!r.ok)throw new Error(data.error||'请求未完成'); return data;
}
function asset(job,kind,sku,face,extra={}) { const params={job,kind,...extra}; if(sku)params.sku=sku;if(face)params.face=face;return '/api/artifact?'+new URLSearchParams(params); }
function seconds(n) { n=Math.round(n||0);return n<60?n+' 秒':Math.floor(n/60)+' 分 '+n%60+' 秒'; }
function updateConcurrency(){const value=$('concurrency').value;$('concurrencyValue').textContent=value;$('concurrency').setAttribute('aria-valuetext',value+' 并发');}
$('concurrency').value=localStorage.getItem('sixview-concurrency')||$('concurrency').value;$('concurrency').oninput=()=>{localStorage.setItem('sixview-concurrency',$('concurrency').value);updateConcurrency();};updateConcurrency();
function updateStart(){const running=jobs.some(j=>active.has(j.status));$('shutdown').disabled=running;$('shutdown').title=running?'请先停止本批再退出':'停止本地网页服务';$('start').disabled=busy||running||!scanData||!selected.size||!selectedModel();$('start').textContent=running?'处理中…':selected.size?`开始处理 · ${selected.size} 台`:'开始处理';$('selectionCount').textContent=`已选 ${selected.size} 台`;$('selectAll').checked=!!scanData&&selected.size===scanData.valid&&selected.size>0;$('selectAll').indeterminate=selected.size>0&&selected.size<scanData.valid;}
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
async function startBatch(skus=Array.from(selected)){busy=true;updateStart();try{const j=await api('start',{scan_id:scanData.id,model:selectedModel(),skus,concurrency:+$('concurrency').value,output:$('output').value.trim()});selectedJob=j.id;localStorage.setItem('sixview-job',j.id);galleryKey='';await refresh();$('jobPanel').scrollIntoView({behavior:'smooth',block:'start'});}catch(e){notice(e.message,true);}finally{busy=false;updateStart();}}
setModel(localStorage.getItem('sixview-model')||'');
$('model').onchange=()=>{localStorage.setItem('sixview-model',selectedModel());updateStart();};
$('start').onclick=()=>startBatch();
function stageText(r){if(['review','complete'].includes(r.status))return '已生成 · '+(r.stage_summary||'处理阶段待核对')+' · '+seconds(r.seconds);if(r.status==='failed')return r.error_hint||r.error||'处理失败，请查看日志';if(r.status==='waiting')return '排队中';if(r.status==='unfinished')return '未完成';return ({loading:'准备中',geometry:'抠图摆正',geometry_resume:'校验并复用几何结果',lighting:'优化光照',alignment:'结构对齐',checking:'检查保存',preserving:'保存已完成阶段并记录待核对项'}[r.stage]||'正在处理')+(r.face?' · '+labels[r.face]:'');}
function faceText(row,face){const stage=row?.face_processing?.[face],lighting=row?.per_face_stages?.[face]?.lighting;const label={geometry:'几何已完成，未应用调光',geometry_source:'几何已完成，未应用调光',lighting:'已调光',verified_lighting:'已调光',independently_verified_lighting:'已调光',pixel_verified_lighting_effect_warning:'已调光',lighting_review:'已应用调光候选，待核对',structure_review:'已调光并调整天线位置，待核对',structure:lighting==='failed'?'天线已对齐，调光未完成':'天线已对齐',seam:'调光与接缝已处理'}[stage]||'该面处理阶段待核对';return label;}
function currentJob(){return jobs.find(j=>j.id===selectedJob);}
function drawTaskRows(job) {
 const container = $('taskRows');
 if (container.dataset.job !== job.id) {
  container.replaceChildren();
  container.dataset.job = job.id;
 }
 const existing = new Map(Array.from(container.children, row => [row.dataset.name, row]));
 const names = new Set(job.rows.map(row => row.name));
 for (const row of Array.from(container.children)) {
  if (!names.has(row.dataset.name)) row.remove();
 }
 job.rows.forEach((result, index) => {
  const previous = existing.get(result.name);
  const signature = JSON.stringify(result);
  let row = previous;
  if (!previous || previous.dataset.signature !== signature) {
   row = node('div', undefined, 'task-row ' + result.status);
   row.dataset.name = result.name;
   row.dataset.signature = signature;
   const description = node('div', undefined, 'description');
   description.append(node('span', stageText(result)));
   if (result.review_reasons?.length) {
    const details = node('details', undefined, 'review-reasons');
    details.open = previous?.querySelector('details')?.open || false;
    details.append(node('summary', result.status === 'complete' ? '查看处理提示' : '查看待核对原因'));
    for (const reason of result.review_reasons) {
     const text = (labels[reason.face] ? labels[reason.face] + '：' : '') + (reason.message || '请核对过程记录');
     const item = node('p', text);
     if (reason.detail && reason.detail !== reason.message) item.append(node('small', reason.detail));
     details.append(item);
    }
    description.append(details);
   }
   if (result.review_continuation_ready && result.status === 'review') {
    description.append(node('small', '处理候选已生成，需人工核对；接缝处理未完成。'));
   } else if (result.review_only && result.status === 'review') {
    description.append(node('small', '处理步骤已完成，请复核候选效果。'));
   }
   if (result.portable_review_available) {
    const bundle = node('a', '带复核记录下载');
    bundle.href = asset(job.id, 'bundle', result.name, null, {download: 1});
    description.append(bundle);
   }
   row.append(node('span', undefined, 'state-dot'), node('span', result.name), description);
   if (previous) previous.replaceWith(row);
  }
  if (container.children[index] !== row) {
   container.insertBefore(row, container.children[index] || null);
  }
 });
}
function drawJob(){
 const j=currentJob();$('jobPanel').hidden=!j;if(!j){$('resultPanel').hidden=true;return;}
 $('jobTitle').textContent=active.has(j.status)?'处理中':'本批结果';if($('taskDetails').dataset.job!==j.id){$('taskDetails').open=active.has(j.status);$('taskDetails').dataset.job=j.id;}$('jobBadge').textContent=statusLabels[j.status]||j.status;$('jobBadge').className='badge push '+(j.status==='complete'?'success':['review','partial','failed','stopped','interrupted'].includes(j.status)?'warn':'');
 $('jobId').textContent=new Date(j.created*1000).toLocaleString('zh-CN',{month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit'});$('jobId').title=j.id;$('jobConcurrency').textContent=`${j.total} 台 · ${j.concurrency} 并发${j.model==='iphone17promax'?' · Pro Max':j.model==='iphone17pro'?' · Pro':''}`;$('jobMessage').hidden=active.has(j.status)||j.status==='complete';$('jobMessage').textContent=j.message;$('doneCount').textContent=`${j.completed} / ${j.total}`;$('elapsed').textContent=seconds(j.elapsed);$('failedCount').textContent=j.failed;$('failedStat').hidden=!j.failed;$('progress').value=j.progress;$('progress').hidden=!active.has(j.status);
 drawTaskRows(j);
 $('stop').hidden=!active.has(j.status);$('stop').disabled=['stopping','packaging'].includes(j.status);$('retry').hidden=active.has(j.status)||!j.retry_count;$('retry').textContent=`继续未完成步骤 · ${j.retry_count||0} 台`;$('retry').disabled=jobs.some(x=>active.has(x.status));$('log').href=asset(j.id,'log');
 $('resultPanel').hidden=!j.completed;
 if(j.completed){$('overview').hidden=!j.downloadable;$('download').hidden=!j.downloadable;$('overview').href=asset(j.id,'overview');$('download').href=asset(j.id,'zip',null,null,{download:1});drawGallery(j);}
}
function drawGallery(j){const key=j.id+'|'+currentFace+'|'+j.rows.filter(r=>r.preview).map(r=>r.name+r.stage_summary).join(',');if(key===galleryKey)return;galleryKey=key;$('gallery').replaceChildren();for(const r of j.rows.filter(x=>x.preview)){const card=node('button',undefined,'image-card'),wrap=node('div',undefined,'image-wrap checker'),img=node('img');img.loading='lazy';img.src=asset(j.id,'image',r.name,currentFace,{thumb:1});img.alt=r.name+' '+labels[currentFace]+' · '+faceText(r,currentFace);wrap.append(img);const caption=node('div',undefined,'caption');caption.append(node('strong',r.name+(r.status==='review'?(r.review_continuation_ready?' · 已处理 · 待核对':r.review_only?' · 效果待核对':' · 步骤未完成'):'')),node('span','↗'));card.append(wrap,caption);card.onclick=()=>showViewer(j,r.name,currentFace);$('gallery').append(card);}}
for(const f of faces){const b=node('button',labels[f],f===currentFace?'selected':'');b.onclick=()=>{currentFace=f;for(const c of $('faceTabs').children)c.classList.toggle('selected',c===b);const j=currentJob();if(j)drawGallery(j);};$('faceTabs').append(b);}
function showViewer(j,sku,face){viewerData={id:j.id,sku,face};const row=j.rows.find(r=>r.name===sku);$('viewerTitle').textContent=sku+' · '+labels[face]+' · '+faceText(row,face);$('imageStage').classList.remove('actual');$('fit').classList.add('selected');$('actual').classList.remove('selected');viewerImage('image');$('downloadImage').href=asset(j.id,'image',sku,face,{download:1});$('viewer').showModal();}
function viewerImage(kind){const d=viewerData;if(!d)return;$('viewerImage').src=asset(d.id,kind,d.sku,d.face);$('showResult').classList.toggle('selected',kind==='image');$('showSource').classList.toggle('selected',kind==='source');}
$('showResult').onclick=()=>viewerImage('image');$('showSource').onclick=()=>viewerImage('source');$('closeViewer').onclick=()=>$('viewer').close();$('background').onclick=()=>$('imageStage').classList.toggle('dark');
$('fit').onclick=()=>{$('imageStage').classList.remove('actual');$('fit').classList.add('selected');$('actual').classList.remove('selected');};$('actual').onclick=()=>{$('imageStage').classList.add('actual');$('actual').classList.add('selected');$('fit').classList.remove('selected');};
$('stop').onclick=async()=>{const j=currentJob();if(!j)return;try{await api('cancel',{id:j.id});await refresh();}catch(e){notice(e.message,true);}};
$('openFolder').onclick=async()=>{try{await api('open',{id:selectedJob});}catch(e){notice(e.message,true);}};
$('retry').onclick=async()=>{const j=currentJob();if(!j)return;$('retry').disabled=true;try{const next=await api('retry',{id:j.id,concurrency:1});selectedJob=next.id;localStorage.setItem('sixview-job',next.id);galleryKey='';await refresh();notice('已创建续接批次；校验后复用几何结果，继续未完成步骤。');}catch(e){notice(e.message,true);$('retry').disabled=false;}};
let historyKey='';
function drawHistory(){const key=jobs.map(j=>j.id+j.status+j.completed).join('|')+selectedJob;if(key===historyKey)return;historyKey=key;if(!jobs.length)return;$('history').replaceChildren(...jobs.slice(0,20).map(j=>{const b=node('button',undefined,'history-item'+(j.id===selectedJob?' selected':'')),info=node('span',undefined,'history-name');info.append(document.createTextNode(j.source.split('/').pop()+' · '+j.total+' 台'),node('small',new Date(j.created*1000).toLocaleString('zh-CN')+' · '+j.concurrency+' 并发'));b.append(info,node('span',statusLabels[j.status]||j.status,'badge'));b.onclick=()=>{selectedJob=j.id;localStorage.setItem('sixview-job',j.id);galleryKey='';drawJob();drawHistory();$('jobPanel').scrollIntoView({behavior:'smooth'});};return b;}));}
async function refresh(){try{const state=await api('state?compact=1');jobs=state.jobs;if(!$('output').value)$('output').value=state.default_output;const m=state.host;$('available').textContent=(m.available/1024**3).toFixed(1)+' GB';$('memoryTotal').textContent=(m.total/1024**3).toFixed(0)+' GB';$('memoryBadge').textContent=m.pressure>=4?'压力较高':m.pressure===2?'稍有压力':'正常';$('memoryBadge').className='badge '+(m.pressure>=2?'warn':'success');$('connectionDot').classList.remove('offline');$('connectionText').textContent='本地运行';if(!jobs.find(j=>j.id===selectedJob))selectedJob=jobs[0]?.id;drawJob();drawHistory();updateStart();}catch(e){$('connectionDot').classList.add('offline');$('connectionText').textContent='未连接，请重新启动';$('start').disabled=true;}}
async function poll(){await refresh();setTimeout(poll,1500);}poll();

$('shutdown').onclick=async()=>{try{await api('shutdown',{});notice('服务已退出');}catch(e){notice(e.message,true);}};
