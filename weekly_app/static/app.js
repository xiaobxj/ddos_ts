const $ = id => document.getElementById(id);
const token = location.hash.slice(1) || sessionStorage.getItem('weekly-app-token');
if(token)sessionStorage.setItem('weekly-app-token',token);
history.replaceState(null, '', location.pathname);
const histories = [['rolling5_annual20','固定5年 · 年度更新'],['weekly_state_validated','季度市场状态修正'],['annual_head_timeweight2y','近期样本时间加权'],['annual_head_weighted_intercept','加权截距混合'],['annual_head_weighted_slopes','加权斜率混合']];
const methods = [['learned_market','市场特征'],['learned_vol_interaction','波动率交互'],['learned_order_extension','涨跌顺序扩展'],['learned_order_offset','涨跌顺序修正']];
const coverage = {COMPLETE:['已成熟',''],PENDING_LABEL:['等待标签','warn'],NOT_DUE:['尚未到期','neutral'],NO_TIMELY_PREDICTION:['未及时记录','bad'],INVALID_LABEL:['标签无效','bad']};
const stages = [['validate','数据与参数'],['fetch','行情采集'],['refresh','到期更新'],['record','本周预测'],['settle','标签结算'],['evaluate','结果汇总']];
const states = {negative_low:'负向趋势 / 低波动',negative_high:'负向趋势 / 高波动',nonnegative_low:'非负趋势 / 低波动',nonnegative_high:'非负趋势 / 高波动'};
let state = null, data = null, lastRevision = -1, stopped = false, logOpen = false, currentTab = 'forecast', localPending = false;
const esc = v => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const pct = v => Number.isFinite(v) ? `${(v*100).toFixed(1)}%` : '—';
const timeText = t => t ? new Date(t).toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}) : '—';
const chip = (text, cls='neutral') => `<span class="chip ${cls}">${esc(text)}</span>`;
function dateCST(){return new Intl.DateTimeFormat('en-CA',{timeZone:'Asia/Shanghai',year:'numeric',month:'2-digit',day:'2-digit'}).format(new Date());}
function empty(title, text){return `<div class="empty"><div class="empty-icon">▥</div><h4>${esc(title)}</h4><p>${esc(text)}</p></div>`;}
async function api(path, method='GET'){
  const response = await fetch(`/api/${path}`,{method,headers:{'X-App-Token':token},cache:'no-store'});
  const value = await response.json();
  if(!response.ok) throw new Error(value.error || `连接失败 (${response.status})`);
  return value;
}
function showNotice(text, cls=''){$('notice').className=`notice ${cls}`;$('notice').textContent=text;}
function friendlyError(text){
  if(/File exists|另一个程序|cycle.lock|writer.lock/i.test(text))return '已有流程占用运行锁。请等待原任务完成；如果程序曾异常退出，请检查运行日志。';
  if(/revised|omitted|changed.*prefix|overlap/i.test(text))return '行情源与已存档的数据不一致，流程已停止，原记录已保留。详情可在运行日志中查看。';
  if(/connection|timeout|resolve|HTTPError|HTTPSConnection|网络/i.test(text))return '行情连接未成功，请检查网络后在当晚记录窗口内重试。';
  if(/package|parameter|frozen|freeze/i.test(text))return '数据或参数检查未通过，流程已停止。请展开日志查看具体原因。';
  return `运行未完成：${text}`;
}
function renderHero(){
  if(!data)return;
  const plan=data.plan, today=dateCST(), local=new Date().toLocaleTimeString('en-GB',{timeZone:'Asia/Shanghai',hour12:false}), hour=Number(local.slice(0,2));
  const isFriday=new Date(`${today}T12:00:00+08:00`).getUTCDay()===5;
  const row=data.weeks.find(w=>w.date===today), recorded=Boolean(data.forecasts[today]);
  let tag='按固定计划持续记录',title='',message='';
  if(recorded){tag='本周预测已经提交';title='这一周，已记录。';message='可在下方查看各方案结果。再次运行不会重复提交，未成熟标签会继续等待后续数据。';}
  else if(isFriday && row && hour>=18 && hour<23){tag='本周记录窗口已开启';title='可以记录本周预测了';message='点击右侧按钮采集完整日线。数据检查与到期参数更新完成后，会提交本周预测。';}
  else if(isFriday && row && hour<18){tag='等待正式记录时段';title='今天18点后，再记录本周';message='现在可以查看已有记录。正式预测需等北京时间18点后，并在23点前完成写入。';}
  else if(isFriday && row && hour>=23){tag='本周记录窗口已结束';title='本周未及时记录';message='此时不再补记预测。仍可运行流程更新数据和历史标签，下一个周五继续。';}
  else if(plan.next_unrecorded_signal){tag=plan.recorded_predictions===0?'等待首个周五记录':'等待下一次周五记录';title=`下一次：${plan.next_unrecorded_signal.slice(5).replace('-','月')}日 · 周五`;message='每周五18点后、23点前运行一次。其他工作日也可更新数据，检查旧预测的成熟结果。';}
  else{tag='固定观察期已结束';title='继续等待标签与最终评估';message='可运行流程更新历史标签和固定评估。观察期以外不会增加新的周预测。';}
  $('hero-tag').textContent=tag;$('hero-title').textContent=title;$('hero-message').textContent=message;
  $('stat-data').textContent=plan.latest_snapshot_end || '暂无';$('stat-pred').innerHTML=`${plan.recorded_predictions}<em>/ 52 周</em>`;
  $('stat-mature').innerHTML=`${data.mature_signals}<em>周</em>`;$('stat-pending').innerHTML=`${plan.pending_labels}<em>周</em>`;
  $('status-updated').textContent=`状态更新于 ${timeText(data.updated_utc)}`;
}
function renderForecast(){
  const date=$('forecast-date').value, forecast=data?.forecasts[date];
  if(!forecast){$('forecast-context').classList.add('hidden');$('forecast-content').innerHTML=empty('还没有真实周预测','首次候选日为2026年9月18日。完成一次有效记录后，各方案结果会显示在这里。');return;}
  const week=data.weeks.find(w=>w.date===date),status=coverage[week.status];
  $('forecast-context').classList.remove('hidden');$('forecast-context').innerHTML=chip(`${date} 记录`)+chip(states[forecast.state]||forecast.state)+chip(status[0],status[1]);
  let html='<div class="table-scroll"><table class="forecast-table"><thead><tr><th>训练与更新方案</th>'+methods.map(m=>`<th>${m[1]}</th>`).join('')+'</tr></thead><tbody>';
  for(const h of histories){html+=`<tr><td>${h[1]}</td>`;for(const m of methods){const r=forecast.ensemble_predictions.find(r=>r.history===h[0]&&r.method===m[0]);html+=r?`<td><span class="probability ${r.probability>.5?'up':'down'}">${pct(r.probability)}</span><span class="direction">方向：${r.probability>.5?'上涨':'非上涨'}</span></td>`:'<td>—</td>'; }html+='</tr>';}
  $('forecast-content').innerHTML=html+'</tbody></table></div>';
}
function renderHistory(){
  if(!data)return;
  const filter=$('history-filter').value, rows=data.weeks.filter(w=>filter==='all'||(filter==='recorded'?Boolean(w.recorded_utc):w.status!=='NOT_DUE'));
  if(!rows.length){$('history-content').innerHTML=empty('这个筛选下暂无记录','新的周预测与成熟标签会在运行后自动更新。');return;}
  let html='<div class="table-scroll"><table><thead><tr><th>周五信号日</th><th>状态</th><th>预测提交时间 · 北京</th><th>目标区间涨跌幅</th><th>结果</th></tr></thead><tbody>';
  for(const w of rows){const [text,cls]=coverage[w.status],r=w.status==='COMPLETE'?w.label?.exec_return:null;html+=`<tr><td>${esc(w.date)}</td><td>${chip(text,cls)}</td><td>${esc(w.recorded_utc?timeText(w.recorded_utc):'—')}</td><td>${Number.isFinite(r)?`${r>0?'+':''}${(r*100).toFixed(2)}%`:'—'}</td><td>${w.recorded_utc?`<button class="table-button" data-week="${esc(w.date)}">查看预测</button>`:'—'}</td></tr>`;}
  $('history-content').innerHTML=html+'</tbody></table></div>';
}
function renderMetrics(){
  if(!data)return;
  $('evaluation-chip').textContent=data.evaluation_status==='PRIMARY_SAVED'?'最终评估已存档':'描述性汇总';
  if(!data.metrics.length){$('metrics-content').innerHTML=empty('还没有成熟样本','预测提交后，需要后续行情完成标签。目前不显示准确率。');return;}
  let html='<div class="table-scroll"><table><thead><tr><th>训练与更新方案</th><th>方法</th><th>成熟周数</th><th>方向正确数</th><th>方向准确率</th><th title="上涨概率与实际0/1结果的均方误差，越低越好">Brier分数 ⓘ</th></tr></thead><tbody>';
  for(const h of histories)for(const m of methods){const r=data.metrics.find(r=>r.history===h[0]&&r.method===m[0]);if(r)html+=`<tr><td>${h[1]}</td><td>${m[1]}</td><td>${r.n}</td><td>${r.correct}</td><td>${pct(r.accuracy)}</td><td>${r.brier.toFixed(4)}</td></tr>`;}
  $('metrics-content').innerHTML=html+'</tbody></table></div>';
}
function renderParameters(){
  const today=dateCST(),p=data.packages.find(p=>p.valid_from<=today&&p.valid_until>=today);
  const items=[['训练记忆','固定5年 · 年度滚动'],['市场状态','季度校验与修正'],['当前参数有效至',p?.valid_until||'等待到期参数更新'],['下一参数截止',data.plan.next_parameter_cutoff||'观察期内更新已完成']];
  $('parameter-content').innerHTML=items.map(([k,v])=>`<div><span>${esc(k)}</span><b>${esc(v)}</b></div>`).join('');
}
function renderProgress(s){
  const relevant=s.job==='cycle';$('progress-panel').classList.toggle('hidden',!relevant);
  if(!relevant)return;
  const map={};for(const l of s.log)if(l.stage)map[l.stage]=l.state;
  if(!s.busy)for(const k of Object.keys(map))if(map[k]==='running')map[k]=s.error?'failed':'done';
  $('progress-title').textContent=s.busy?'正在执行本次流程':s.error?'本次流程未完成':'本次运行已结束';
  $('progress-state').textContent=s.busy?'请保持程序运行':s.error?'查看原因':'已结束';
  $('progress-steps').innerHTML=stages.map(([k,t])=>`<div class="step ${map[k]||''}"><b>${map[k]==='done'?'✓':map[k]==='running'?'◌':map[k]==='failed'?'!':'·'}</b>${t}${!s.busy&&!map[k]?' · 未执行':''}</div>`).join('');
  $('progress-message').textContent=s.busy?(s.log.at(-1)?.message||'准备运行…'):'未执行的步骤表示本次无需处理、未到时点或已停止，请以结果说明为准。';
}
function render(s){
  state=s;$('run-button').disabled=s.busy||localPending;$('refresh-button').disabled=s.busy||localPending;$('quit-button').disabled=s.busy||localPending;
  $('run-label').textContent=s.busy?(s.job==='cycle'?'运行中，请稍候':'正在读取状态'):'运行本周流程';
  renderProgress(s);
  $('log-summary').textContent=s.error?friendlyError(s.error):(s.log.at(-1)?.message||'本次打开仅检查状态。');
  $('log-content').textContent=s.log.map(l=>`${timeText(l.at)}  ${l.message}`).join('\n')+(s.error?`\n错误详情：${s.error}`:'');
  if(s.data&&(s.revision!==lastRevision||!data)){
    const selected=$('forecast-date').value, previousLatest=data?.latest_prediction;data=s.data;lastRevision=s.revision;
    const dates=Object.keys(data.forecasts).sort().reverse();$('forecast-date').innerHTML=dates.length?dates.map(d=>`<option value="${esc(d)}">${esc(d)}</option>`).join(''):'<option value="">暂无记录</option>';
    if(data.latest_prediction&&data.latest_prediction!==previousLatest)$('forecast-date').value=data.latest_prediction;
    else if(dates.includes(selected))$('forecast-date').value=selected;
    renderHero();renderForecast();renderHistory();renderMetrics();renderParameters();
  }
  if(!s.busy){
    if(s.error)showNotice(friendlyError(s.error),'bad');
    else if(s.result){const r=s.result;
      if(r.status==='PRE_CLOSE_WAIT')showNotice('已检查完成：现在尚未到18点，没有采集或提交预测。请在周五18点后、23点前再次运行。','warn');
      else if(r.status==='DATA_PENDING')showNotice('本次仍有数据未到齐，尚未完成全部步骤。请查看周记录；若缺少本周预测，可在当晚23点前重试。','warn');
      else if(r.status==='COMPLETED'){
        const p=r.steps.find(x=>x.action==='record'&&x.sequence),duplicate=r.steps.some(x=>x.status==='ALREADY_RECORDED');
        showNotice(p?`已完成：${p.date} 的预测已保存，历史标签也已检查。`:duplicate?'已完成：本周已有预测，本次没有重复提交；数据与标签已检查。':'已完成数据与标签检查。本次不在可记录的周五时段，没有新增周预测。');
      }
    }
  }
}
async function poll(){
  if(stopped)return;
  try{const s=await api('state');if(stopped)return;render(s);}catch(error){if(stopped)return;showNotice(`本地程序未连接：${error.message}。可重新双击启动入口。`,'bad');$('run-button').disabled=true;$('refresh-button').disabled=true;}
  if(!stopped)setTimeout(poll,state?.busy?1000:4000);
}
async function start(action){
  if(localPending||state?.busy)return;
  localPending=true;$('run-button').disabled=true;$('refresh-button').disabled=true;
  try{await api(action,'POST');$('notice').classList.add('hidden');render(await api('state'));}catch(e){showNotice(e.message,'bad');}
  finally{localPending=false;if(state)render(state);}
}
function setTab(tab){currentTab=tab;for(const name of ['forecast','history','metrics']){$(`view-${name}`).classList.toggle('hidden',name!==tab);$(`tab-${name}`).classList.toggle('active',name===tab);$(`tab-${name}`).setAttribute('aria-selected',String(name===tab));}}
$('run-button').addEventListener('click',()=>start('cycle'));
$('refresh-button').addEventListener('click',()=>start('inspect'));
$('forecast-date').addEventListener('change',renderForecast);$('history-filter').addEventListener('change',renderHistory);
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>setTab(b.dataset.tab)));
$('history-content').addEventListener('click',e=>{const b=e.target.closest('[data-week]');if(b){$('forecast-date').value=b.dataset.week;setTab('forecast');renderForecast();$('tab-forecast').focus();}});
$('log-button').addEventListener('click',()=>{logOpen=!logOpen;$('log-content').classList.toggle('hidden',!logOpen);$('log-button').textContent=logOpen?'收起详情':'展开详情';$('log-button').setAttribute('aria-expanded',String(logOpen));});
$('help-button').addEventListener('click',()=>$('help-dialog').showModal());$('close-help').addEventListener('click',()=>$('help-dialog').close());
$('export-button').addEventListener('click',()=>{
  if(!data)return;
  const rows=[['周五信号日','状态','提交时间_北京时间','目标区间涨跌幅','入场日','退出日','联合成熟日'],...data.weeks.map(w=>[w.date,coverage[w.status][0],w.recorded_utc?timeText(w.recorded_utc):'',w.status==='COMPLETE'?(w.label?.exec_return??''):'',w.label?.entry_date??'',w.label?.exit_date??'',w.label?.joint_completed??''])];
  const csv='\ufeff'+rows.map(r=>r.map(v=>`"${String(v).replaceAll('"','""')}"`).join(',')).join('\r\n');
  const url=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'})),a=document.createElement('a');a.href=url;a.download=`每周预测记录_${dateCST()}.csv`;a.click();setTimeout(()=>URL.revokeObjectURL(url),5000);
});
$('quit-button').addEventListener('click',async()=>{try{await api('quit','POST');stopped=true;document.body.innerHTML='<main class="stopped"><span class="eyebrow">SESSION CLOSED</span><h1>本次已退出，记录已保留。</h1><p class="muted">可以关闭这个标签页，下周双击「打开每周预测.cmd」继续。</p></main>';}catch(e){showNotice(e.message,'warn');}});
setInterval(()=>{if(stopped)return;$('clock')&&($('clock').textContent=new Date().toLocaleString('zh-CN',{timeZone:'Asia/Shanghai',hour12:false}));if(data&&!state?.busy)renderHero();},1000);
if(!token){showNotice('请关闭此页，双击「打开每周预测.cmd」重新打开。','warn');}else{poll();}
