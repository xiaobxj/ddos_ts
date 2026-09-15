// Test UI state transitions without a connected browser or third-party DOM package.
const assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),vm=require('node:vm');
class Element{
  constructor(id){this.id=id;this.textContent='';this.disabled=false;this.value='';this.listeners={};this.attrs={};this.dataset={};this.classes=new Set();this._html='';
    this.classList={add:(x)=>this.classes.add(x),remove:x=>this.classes.delete(x),toggle:(x,on)=>{if(on===undefined)on=!this.classes.has(x);on?this.classes.add(x):this.classes.delete(x);}};
  }
  set innerHTML(value){this._html=value;if(this.id==='forecast-date')this.value=value.match(/value="([^"]*)"/)?.[1]||'';}
  get innerHTML(){return this._html;}
  set className(value){this.classes=new Set(value.split(' '));}
  get className(){return [...this.classes].join(' ');}
  setAttribute(k,v){this.attrs[k]=v;}
  addEventListener(name,fn){this.listeners[name]=fn;}
  focus(){} click(){this.listeners.click?.({target:this});} showModal(){this.open=true;} close(){this.open=false;}
}
const elements=new Map();
const get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
for(const name of ['forecast','history','metrics'])get('tab-'+name).dataset.tab=name;
get('history-filter').value='all';
const store=new Map(),intervals=[];
const context=vm.createContext({document:{getElementById:get,querySelectorAll:()=>['forecast','history','metrics'].map(n=>get('tab-'+n)),createElement:()=>new Element('a'),body:new Element('body')},
  location:{hash:'',pathname:'/'},history:{replaceState(){}},sessionStorage:{getItem:k=>store.get(k),setItem:(k,v)=>store.set(k,v)},
  setInterval:fn=>intervals.push(fn),setTimeout:()=>0,fetch:async()=>{throw new Error('Unexpected network in isolated UI test');},
  console,Intl,Date,Number,String,Object,Boolean,JSON,Blob,URL});
const source=fs.readFileSync(path.join(__dirname,'static/app.js'),'utf8');vm.runInContext(source,context);
const run=code=>vm.runInContext(code,context);
const fixture=JSON.parse(fs.readFileSync(path.join(__dirname,'../_work/weekly_app_ui_fixture.json'),'utf8'));
function state(data,revision,result=null,busy=false,error=null){return {data,revision,result,busy,error,job:'cycle',log:[]};}
function render(s){context.input=s;run('render(input)');}
const checks=[];function pass(n){checks.push(n);}

const empty=structuredClone(fixture);empty.forecasts={};empty.metrics=[];empty.latest_prediction=null;empty.mature_signals=0;empty.plan.recorded_predictions=0;empty.plan.pending_labels=0;
render(state(empty,1));assert.match(get('forecast-content').innerHTML,/还没有真实周预测/);assert.match(get('metrics-content').innerHTML,/还没有成熟样本/);pass('empty_state_does_not_show_zero_accuracy');
render(state(fixture,2));assert.equal((get('forecast-content').innerHTML.match(/class="probability /g)||[]).length,20);assert.equal(get('forecast-date').value,'2026-09-18');assert.doesNotMatch(get('forecast-content').innerHTML,/NaN|undefined/);pass('all20_probabilities_and_latest_date_render');
assert.match(get('history-content').innerHTML,/未及时记录/);assert.equal((get('history-content').innerHTML.match(/<tr>/g)||[]).length,53);pass('all52_slots_and_missing_friday_visible');
get('history-filter').value='recorded';run('renderHistory()');assert.equal((get('history-content').innerHTML.match(/<tr>/g)||[]).length,2);pass('recorded_filter_selects_only_timely_predictions');
run("setTab('metrics')");assert.equal(get('tab-metrics').attrs['aria-selected'],'true');assert.equal(get('view-metrics').classes.has('hidden'),false);assert.equal((get('metrics-content').innerHTML.match(/<tr>/g)||[]).length,21);pass('metrics_tab_has20_common_sample_rows');
const next=structuredClone(fixture);next.latest_prediction='2026-09-25';next.forecasts['2026-09-25']=structuredClone(next.forecasts['2026-09-18']);next.forecasts['2026-09-25'].signal_date='2026-09-25';next.weeks[1].status='PENDING_LABEL';next.weeks[1].recorded_utc='2026-09-25T10:30:00+00:00';
render(state(next,3));assert.equal(get('forecast-date').value,'2026-09-25');pass('new_week_automatically_selects_new_prediction');
render(state(next,4,null,true));assert.equal(get('run-button').disabled,true);assert.equal(get('quit-button').disabled,true);assert.match(get('run-label').textContent,/运行中/);pass('running_disables_repeat_submission_and_quit');
render(state(next,5,{status:'PRE_CLOSE_WAIT',steps:[]}));assert.match(get('notice').textContent,/尚未到18点/);pass('preclose_gives_specific_nonrecording_reason');
render(state(next,6,{status:'COMPLETED',steps:[{action:'record',status:'ALREADY_RECORDED'}]}));assert.match(get('notice').textContent,/没有重复提交/);pass('duplicate_run_message_is_unambiguous');
render(state(next,7,null,false,'Snapshot revised or omitted the recorded prefix'));assert.match(get('notice').textContent,/原记录已保留/);pass('changed_prefix_failure_has_actionable_message');
get('help-button').click();assert.equal(get('help-dialog').open,true);get('close-help').click();assert.equal(get('help-dialog').open,false);pass('help_open_close_handlers');
const reloadContext=vm.createContext({...context,location:{hash:'#fixture-token',pathname:'/'},fetch:()=>new Promise(()=>{})});vm.runInContext(source,reloadContext);assert.equal(store.get('weekly-app-token'),'fixture-token');pass('session_token_saved_for_browser_reload');
run('stopped=true');for(const callback of intervals)callback();pass('quit_stops_periodic_rendering');
const result={status:'PASS',checks,scope:'Node VM with DOM stubs; not a real-browser visual or end-to-end test'};
fs.writeFileSync(path.join(__dirname,'../_work/weekly_app_frontend_verification.json'),JSON.stringify(result,null,2));console.log(JSON.stringify({status:'PASS',checks:checks.length}));
