import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

// Usage: node export_workbook.mjs [Q2 project directory]
// Requires @oai/artifact-tool; the numerical model itself runs entirely in Python.
const root=process.argv[2] || process.cwd();
const data=JSON.parse(await fs.readFile(path.join(root,'results/workbook_data.json'),'utf8'));
const pricesText=(await fs.readFile(path.join(root,'input/q2_fixed_prices.csv'),'utf8')).replace(/^\uFEFF/,'');
const prices=pricesText.trim().split(/\r?\n/).slice(1).map(r=>Number(r.split(',')[1]));
const wb=Workbook.create();
const plan=wb.worksheets.add('计划购电量');
const storage=wb.worksheets.add('充放电量');
const emerg=wb.worksheets.add('紧急购电量');
const costs=wb.worksheets.add('费用汇总');
const notes=wb.worksheets.add('口径说明');
const previewDir=path.join(root,'verification/workbook_previews');
await fs.mkdir(previewDir,{recursive:true});
function letter(n){let s='';for(;n;n=Math.floor((n-1)/26))s=String.fromCharCode(65+(n-1)%26)+s;return s;}
function date(v){return new Date(v+'T00:00:00Z');}
function table(sheet,rows,widths){
  const nr=rows.length,nc=rows[0].length;
  sheet.getRangeByIndexes(0,0,nr,nc).values=rows;
  sheet.showGridLines=false;
  const used=sheet.getRangeByIndexes(0,0,nr,nc);
  used.format.font={name:'Arial',size:10};used.format.rowHeight=23;
  used.format.verticalAlignment='center';
  sheet.getRangeByIndexes(0,0,1,nc).format={fill:'#244A73',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},rowHeight:32,horizontalAlignment:'center'};
  for(let c=0;c<nc;c++)sheet.getRangeByIndexes(0,c,nr,1).format.columnWidth=widths[c] || 17;
  sheet.getRangeByIndexes(1,0,nr-1,1).setNumberFormat('yyyy-mm-dd');
  sheet.freezePanes.freezeRows(1);
}

const pr=data.planned_purchase_wide;
const rows=[['日期\\时间',...pr.columns.slice(1,145),'全天计划购电量 (kWh)','全天计划购电费 (元)'],...pr.data.map(r=>[date(r[0]),...r.slice(1)])];
table(plan,rows,[15]);
plan.getRangeByIndexes(1,1,334,146).setNumberFormat('#,##0.0000');
plan.getRangeByIndexes(0,145,335,2).format.columnWidth=24;
plan.freezePanes.freezeColumns(1);
for(let r=2;r<=335;r++){
  plan.getRange(`${letter(146)}${r}`).formulas=[[`=SUM(B${r}:${letter(145)}${r})`]];
  plan.getRange(`${letter(147)}${r}`).formulas=[[`=SUMPRODUCT(B${r}:${letter(145)}${r},'口径说明'!$B$20:$${letter(145)}$20)`]];
}
const srows=[['日期','时间段','充电量 (kWh)','放电量 (kWh)','时刻','储电量 (kWh)']];
for(let i=0;i<data.storage_4h.data.length;i++){
  const [day,interval,c,d,e0,e1]=data.storage_4h.data[i];let instant=null,e=null;
  if(i%6===0){instant='00:00';e=e0;}
  if(i%6===1){instant='24:00';e=data.storage_4h.data[i+4][5];}
  srows.push([date(day),interval,c,d,instant,e]);
}
table(storage,srows,[15,19,21,21,12,22]);
storage.getRange(`C2:D${srows.length}`).setNumberFormat('#,##0.0000');
storage.getRange(`F2:F${srows.length}`).setNumberFormat('#,##0.0000');
const events=new Map();for(const r of data.emergency_events.data){if(!events.has(r[0]))events.set(r[0],[]);events.get(r[0]).push(r);}
const erows=[['日期','购电时间段','紧急购电量 (kWh)']];
for(const row of data.daily_summary.data){
  const day=row[0],evt=events.get(day)||[];
  if(evt.length===0)erows.push([date(day),'无',0]);
  else for(const r of evt)erows.push([date(day),r[1],r[2]]);
}
table(emerg,erows,[15,24,25]);
emerg.getRange(`C2:C${erows.length}`).setNumberFormat('#,##0.0000');
const dc=data.daily_summary.columns;const ix=Object.fromEntries(dc.map((x,i)=>[x,i]));
const crows=[['日期','计划购电量 (kWh)','应急购电量 (kWh)','计划费用 (元)','应急费用 (元)','总费用 (元)','00:00 电量 (kWh)','24:00 电量 (kWh)','未利用供电 (kWh)']];
for(const d of data.daily_summary.data)crows.push([date(d[0]),d[ix.planned_grid_kwh],d[ix.emergency_kwh],d[ix.planned_cost_yuan],d[ix.emergency_cost_yuan],d[ix.total_cost_yuan],d[ix.initial_storage_kwh],d[ix.terminal_storage_kwh],d[ix.unused_energy_kwh]]);
table(costs,crows,[15,25,25,24,24,24,25,25,25]);
costs.getRange('B2:I335').setNumberFormat('#,##0.0000');
for(let r=2;r<=335;r++){
 costs.getRange(`B${r}`).formulas=[[`='计划购电量'!${letter(146)}${r}`]];
 costs.getRange(`D${r}`).formulas=[[`='计划购电量'!${letter(147)}${r}`]];
 costs.getRange(`F${r}`).formulas=[[`=D${r}+E${r}`]];
}
costs.getRange('A337').values=[['2—12 月累计']];
for(const c of ['B','C','D','E','F','I'])costs.getRange(`${c}337`).formulas=[[`=SUM(${c}2:${c}335)`]];
costs.getRange('B337:I337').setNumberFormat('#,##0.0000');
costs.getRange('A337:I337').format.font={name:'Arial',size:10,bold:true};
notes.showGridLines=false;
notes.getRange('A1:B14').values=[
 ['第二问结果说明','时间口径暂定，正式提交前请确认竞赛方解释。'],
 ['结果方法','28 日历史情景日前优化，普通购电固定，电池按当期供需实时执行。'],
 ['数据来源','附件 1：分时电价。附件 2：负载与光伏实际功率。题目：问题 2 与附录 1、2。'],
 ['评价范围','2025-02-01 至 2025-12-31，共 334 天、48,096 个十分钟时段。'],
 ['时间解释','功率样本时刻 t 解释为此前 10 分钟平均功率，kWh = kW / 6。'],
 ['模板差异','原模板第一列 0:10-0:20；本表第一列 00:00-00:10。最后一列为 23:50-24:00。'],
 ['费用结算','计划费用按全部计划量收取；应急费用按当时附件 1 电价的 5 倍计算。费用汇总给出两者之和。'],
 ['储能解释','充、放电效率各取 90%；实际功率 ≤ 5000 kW；电量始终为 1200—10800 kWh。'],
 ['初始化','1 月 1 日初态 6000 kWh，电池待机；1 月 2 日起滚动运行预热，实际状态连续传递。'],
 ['末端目标','计划末态目标 6000 kWh 是策略假设。实际末态随执行变化，次日不重置。'],
 ['充放电与应急表','记录实际执行结果。连续应急时段在同一天内合并，无应急的日期明确记 0。'],
 ['未利用供电','可能含光伏和已购电，不等同于全部弃光，已购部分仍付费。'],
 ['结果更新','此文件为默认参数运行结果。修改 Notebook 后先更新 CSV/JSON，再重新导出本表。'],
 ['适用边界','日前模型对历史情景近似目标求优；实际控制为启发式，全年结果不代表全局最优。']
];
notes.getRange('A1:B14').format.font={name:'Arial',size:11};
notes.getRange('A1:A14').format.columnWidth=23;
notes.getRange('B1:B14').format.columnWidth=112;
notes.getRange('A1:B14').format.rowHeight=34;
notes.getRange('B1:B14').format.wrapText=true;
notes.getRange('A1:B1').format.font={name:'Arial',size:12,bold:true,color:'#244A73'};
notes.getRange('A17').values=[['附件 1 固定电价 (元/kWh)']];
notes.getRangeByIndexes(18,0,1,145).values=[['时间段',...pr.columns.slice(1,145)]];
notes.getRangeByIndexes(19,0,1,145).values=[['固定电价',...prices]];
notes.getRangeByIndexes(19,1,1,144).setNumberFormat('0.0000');

wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'费用汇总!A337:I337',include:'values,formulas',tableMaxRows:1,tableMaxCols:9,maxChars:1800})).ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:20},maxChars:1000});
console.log(errors.ndjson);
for(const [sheet,range] of [['计划购电量','A1:H7'],['充放电量','A1:F13'],['紧急购电量','A1:C15'],['费用汇总','A1:F7'],['口径说明','A1:B14']]){
 const blob=await wb.render({sheetName:sheet,range,scale:1.5,format:'png'});
 await fs.writeFile(path.join(previewDir,sheet+'.png'),new Uint8Array(await blob.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(root,'result2_时间口径暂定.xlsx'));
console.log('Exported result2_时间口径暂定.xlsx');
