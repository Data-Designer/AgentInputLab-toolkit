const state = { status: null, version: null, items: [], annotations: new Map(), index: 0 };
const fields = {
  input_audit: [
    "unclear_or_misaligned_request", "decomposition_need", "missing_external_evidence",
    "clarification_need", "context_overload_or_noise", "low_defect_state",
    "reformulate_eligible", "decompose_eligible", "acquire_eligible",
    "clarify_eligible", "curate_eligible"
  ],
  trajectory_audit: ["overall_contract_valid", "entity_preserved", "numeric_preserved", "constraint_preserved", "task_drift"]
};
const zh = {
  unclear_or_misaligned_request:"请求不清晰或表达错位", decomposition_need:"需要任务分解", missing_external_evidence:"缺少外部证据",
  clarification_need:"需要澄清", context_overload_or_noise:"上下文过载或噪声", low_defect_state:"低缺陷输入",
  reformulate_eligible:"可使用 Reformulate", decompose_eligible:"可使用 Decompose", acquire_eligible:"可使用 Acquire",
  clarify_eligible:"可使用 Clarify", curate_eligible:"可使用 Curate", overall_contract_valid:"整体 contract 有效",
  entity_preserved:"实体保持", numeric_preserved:"数值保持", constraint_preserved:"约束保持", task_drift:"发生 task drift"
};

async function api(path, options={}) {
  const response = await fetch(path, {headers:{"Content-Type":"application/json"}, ...options});
  const body = await response.json();
  if (!response.ok) throw new Error(body.error?.message || "请求失败");
  return body;
}
function toast(message) { const el=document.querySelector("#toast"); el.textContent=message; el.classList.add("show"); setTimeout(()=>el.classList.remove("show"),1800); }
function completed(version) { return version.status === "complete"; }

async function loadStatus() {
  state.status = await api("/api/status");
  const select=document.querySelector("#version-select");
  select.innerHTML='<option value="">请选择</option>';
  state.status.versions.forEach(v => {
    const option=document.createElement("option"); option.value=v.version_id;
    option.textContent=`${v.version_id} · ${v.status} · ${v.annotation_count}/${v.item_count}`;
    if (v.version_id === state.status.selected_version) option.textContent += " · 已选";
    select.append(option);
  });
  const message=document.querySelector("#version-message");
  message.textContent=state.status.completed_version_exists
    ? "检测到已完成标注。请明确复用一个完成版本，或新建一个不会覆盖旧结果的重标版本。"
    : "尚无完成标注。可以继续未完成版本，或新建第一个标注版本。";
}

async function openVersion(versionId) {
  const payload=await api(`/api/versions/${versionId}`);
  state.version=payload.manifest; state.items=payload.items;
  state.annotations=new Map(payload.annotations.map(row=>[row.item_id,row]));
  state.index=Math.max(0,state.items.findIndex(item=>!state.annotations.has(item.item_id)));
  if (state.index < 0) state.index=0;
  document.querySelector("#workspace").hidden=completed(state.version);
  document.querySelector("#complete-panel").hidden=completed(state.version) || state.annotations.size !== state.items.length;
  document.querySelector("#version-label").textContent=`${state.version.version_id} · ${state.version.status}`;
  if (completed(state.version)) {
    await api("/api/selection",{method:"POST",body:JSON.stringify({version_id:versionId})});
    toast("已选择完成版本供后续实验与绘图使用"); await loadStatus(); return;
  }
  renderList(); renderItem(); updateProgress();
}

function renderList() {
  const list=document.querySelector("#item-list"); list.innerHTML="";
  state.items.forEach((item,index)=>{
    const button=document.createElement("button"); button.type="button";
    button.className=`item-link${index===state.index?" active":""}${state.annotations.has(item.item_id)?" done":""}`;
    button.innerHTML=`<b>${String(index+1).padStart(3,"0")}</b><span></span><i></i>`;
    button.querySelector("span").textContent=item.item_id;
    button.addEventListener("click",()=>{state.index=index;renderList();renderItem();}); list.append(button);
  });
}
function renderItem() {
  const item=state.items[state.index]; const prior=state.annotations.get(item.item_id);
  document.querySelector("#item-kind").textContent=state.status.kind.replace("_"," · ").toUpperCase();
  document.querySelector("#item-title").textContent=`${state.index+1}. ${item.item_id}`;
  const visible={...item}; delete visible.item_id; delete visible.audit_item_id;
  document.querySelector("#item-content").textContent=JSON.stringify(visible,null,2);
  const container=document.querySelector("#label-fields"); container.innerHTML="";
  fields[state.status.kind].forEach(name=>{
    const row=document.createElement("div"); row.className="label-row";
    const label=document.createElement("label"); label.textContent=zh[name] || name;
    const select=document.createElement("select"); select.name=name; select.required=true;
    const options=state.status.kind==="input_audit" ? ["","yes","no","uncertain"] : ["","yes","no","uncertain"];
    options.forEach(value=>{const option=document.createElement("option");option.value=value;option.textContent=value||"请选择";select.append(option);});
    if (prior) select.value=String(prior.labels[name] ?? "");
    row.append(label,select); container.append(row);
  });
  document.querySelector("#notes").value=prior?.notes || "";
  const status=document.querySelector("#save-state"); status.textContent=prior?"已保存":"未保存"; status.classList.toggle("saved",Boolean(prior));
  document.querySelector("#previous-button").disabled=state.index===0;
}
function updateProgress() {
  const count=state.annotations.size,total=state.items.length;
  document.querySelector("#progress-number").textContent=`${count} / ${total}`;
  document.querySelector("#progress-bar").style.width=`${total?100*count/total:0}%`;
  document.querySelector("#complete-panel").hidden=count!==total;
}

document.querySelector("#new-button").addEventListener("click",async()=>{
  try {
    if (state.status.completed_version_exists && !confirm("已有完成版本。确认新建一个独立的重标版本吗？旧版不会被覆盖。")) return;
    const created=await api("/api/versions",{method:"POST",body:JSON.stringify({action:"new",annotator_id:document.querySelector("#annotator-id").value})});
    await loadStatus(); document.querySelector("#version-select").value=created.version_id; await openVersion(created.version_id);
  } catch(error){toast(error.message);}
});
document.querySelector("#resume-button").addEventListener("click",async()=>{
  const id=document.querySelector("#version-select").value; if(!id){toast("请先选择版本");return;} try{await openVersion(id);}catch(error){toast(error.message);}
});
document.querySelector("#annotation-form").addEventListener("submit",async event=>{
  event.preventDefault(); const item=state.items[state.index]; const labels={};
  for(const select of document.querySelectorAll("#label-fields select")){if(!select.value){toast("请完成所有标签");return;}labels[select.name]=select.value;}
  try {
    await api(`/api/versions/${state.version.version_id}/annotations`,{method:"POST",body:JSON.stringify({item_id:item.item_id,labels,notes:document.querySelector("#notes").value})});
    const payload=await api(`/api/versions/${state.version.version_id}`); state.annotations=new Map(payload.annotations.map(row=>[row.item_id,row]));
    if(state.index<state.items.length-1)state.index+=1; renderList();renderItem();updateProgress();toast("已自动保存");
  } catch(error){toast(error.message);}
});
document.querySelector("#previous-button").addEventListener("click",()=>{if(state.index>0){state.index-=1;renderList();renderItem();}});
document.querySelector("#complete-button").addEventListener("click",async()=>{
  try{await api(`/api/versions/${state.version.version_id}/complete`,{method:"POST",body:"{}"});await api("/api/selection",{method:"POST",body:JSON.stringify({version_id:state.version.version_id})});toast("版本已冻结并选中");await loadStatus();await openVersion(state.version.version_id);}catch(error){toast(error.message);}
});

loadStatus().catch(error=>toast(error.message));
