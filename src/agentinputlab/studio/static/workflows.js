"use strict";
const byId = id => document.getElementById(id);
const state = {catalog: null, inspected: null, selected: null, busy: false, timer: null};
const pretty = value => JSON.stringify(value, null, 2);
function message(text, error = false) {
  byId("message").textContent = text;
  byId("message").classList.toggle("error", error);
  if (error) byId("message").focus();
}
async function api(path, payload) {
  const response = await fetch("/api/workflows/" + path, {
    method: payload === undefined ? "GET" : "POST",
    headers: payload === undefined ? {} : {"Content-Type": "application/json", "X-Studio-Session": state.catalog.session},
    body: payload === undefined ? undefined : JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error?.message || "Workflow request failed");
  return result;
}
function object(id) {
  const field = byId(id);
  try {
    const value = JSON.parse(field.value);
    if (value === null || typeof value !== "object" || Array.isArray(value)) throw new Error("Expected a JSON object");
    field.removeAttribute("aria-invalid");
    return value;
  } catch (error) {
    field.setAttribute("aria-invalid", "true"); field.focus();
    throw new Error(`${field.closest("label")?.childNodes[0]?.textContent.trim() || id}: ${error.message}`);
  }
}
function invalidate(note = "Configuration changed. Check prerequisites before running.") {
  state.inspected = null; byId("launch").disabled = true;
  byId("inspection").replaceChildren(); byId("draft-note").textContent = note;
}
function setDraft(workflow) {
  byId("workflow").value = pretty(workflow);
  byId("export-workflow").disabled = false; byId("export-run").disabled = false;
  byId("inspect").disabled = !state.catalog.execution_enabled;
  invalidate();
}
function recipeNote() {
  const recipe = state.catalog.recipes.find(r => r.release_id === byId("method").value);
  byId("recipe-note").textContent = recipe ? `Route: ${recipe.route}. Required native parameters: ${recipe.required_parameters.join(", ") || "none"}.${recipe.retriever_required ? " A retriever is required." : ""}` : "";
}
function applyPreset() {
  const preset = state.catalog.presets.find(p => p.id === byId("preset").value);
  byId("method").value = preset.release_id;
  byId("benchmark").value = preset.benchmark_id || "";
  byId("query").disabled = Boolean(preset.benchmark_id);
  for (const id of ["parameters", "retriever", "artifacts", "runtime", "preparation", "environment"]) byId(id).value = "{}";
  recipeNote();
  invalidate("Preset selected. Generate a draft to apply these settings.");
}
async function task(action) {
  if (state.busy) return;
  state.busy = true;
  byId("message").setAttribute("aria-busy", "true");
  message("Working…");
  try { await action(); } catch (error) { message(error.message, true); }
  finally { state.busy = false; byId("message").removeAttribute("aria-busy"); }
}
async function generate() {
  if (!byId("builder").reportValidity()) return;
  const request = {
    release_id: byId("method").value, query: byId("query").value,
    limit: Number(byId("limit").value), seed: Number(byId("seed").value),
    max_workers: Number(byId("workers").value),
    budget: {max_output_tokens: Number(byId("tokens").value), max_model_calls: Number(byId("calls").value),
      max_retrieval_calls: Number(byId("retrieval-calls").value), max_tool_calls: Number(byId("tool-calls").value)},
  };
  if (byId("benchmark").value) request.benchmark_id = byId("benchmark").value;
  const model = byId("model").value.trim(), endpoint = byId("endpoint").value.trim(), revision = byId("revision").value.trim();
  if (model || endpoint || revision) {
    if (!model || !endpoint || !revision) throw new Error("Provide model ID, API URL and deployment revision together.");
    request.backend = {model_id: model, base_url: endpoint, deployment_identity: {revision}, api_key_env: byId("key-env").value};
  }
  for (const id of ["parameters", "retriever", "artifacts", "runtime", "preparation", "environment"]) {
    const value = object(id);
    if (Object.keys(value).length) request[id === "environment" ? "environment_backend" : id] = value;
  }
  const generated = await api("template", request);
  setDraft(generated.workflow);
  byId("parameter-reference").textContent = pretty(generated.parameters);
  message("Configuration generated. Review it or check prerequisites; no model calls have been made.");
}
function renderInspection(report) {
  const container = byId("inspection"); container.replaceChildren();
  const title = document.createElement("h3"); title.textContent = report.disposition.replaceAll("_", " "); container.append(title);
  const list = document.createElement("ul");
  for (const check of report.checks || []) {
    const item = document.createElement("li"); item.className = check.status === "blocked" ? "blocked" : "";
    item.textContent = `${check.check}: ${check.status}${check.action ? " — " + check.action : ""}${check.error ? " — " + JSON.stringify(check.error) : ""}`;
    list.append(item);
  }
  container.append(list);
  const details = document.createElement("details"), summary = document.createElement("summary"), pre = document.createElement("pre");
  summary.textContent = "Full preflight report"; pre.textContent = pretty(report); details.append(summary, pre); container.append(details);
}
async function inspect() {
  const workflow = object("workflow"), snapshot = byId("workflow").value;
  const report = await api("inspect", {workflow, probe_endpoint: byId("probe").checked});
  if (snapshot !== byId("workflow").value) { invalidate(); return; }
  renderInspection(report); state.inspected = snapshot;
  byId("launch").disabled = !(report.can_attempt || report.can_prepare);
  message(report.can_attempt ? "Local prerequisites checked. Execution will check the model endpoint before dispatch." : report.can_prepare ? "Preparation is required. Review the report and download setting before starting." : "Resolve the listed prerequisites, then check again.");
}
function download(value, filename) {
  const url = URL.createObjectURL(new Blob([pretty(value) + "\n"], {type:"application/json"}));
  const link = document.createElement("a"); link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
async function refreshRuns() {
  if (!state.catalog.execution_enabled) return;
  const result = await api("runs"); byId("runs").replaceChildren();
  for (const run of result.runs) {
    const button = document.createElement("button"); button.type = "button";
    button.textContent = `${run.status} · ${run.release_id} · ${run.id.slice(0, 8)}`;
    button.addEventListener("click", () => task(() => showRun(run.id))); byId("runs").append(button);
  }
  if (!result.runs.length) byId("runs").textContent = "No saved workflows yet.";
}
async function showRun(id) {
  const run = await api("runs/" + encodeURIComponent(id)); state.selected = run;
  window.history.replaceState(null, "", "#run=" + run.id);
  byId("run-detail").hidden = false;
  byId("run-title").textContent = `${run.status} · ${run.id.slice(0, 8)}`;
  byId("output-path").textContent = "Server output: " + run.output_dir;
  const result = run.result || {};
  byId("run-summary").textContent = pretty({status: run.status, error: run.error,
    terminal: result.terminal_count, selected: result.selected_count, outcomes: result.outcomes,
    automatic_retry: false});
  byId("run-report").textContent = pretty(result);
  byId("run-results").textContent = pretty((run.preview?.records || []).map(row => ({
    case_id: row.workflow_case_id, status: row.status, scored: row.scored,
    metrics: row.evaluation?.metrics || row.metrics,
    usage: row.usage || row.output?.record?.usage,
    failure: row.failure,
    output: row.output?.state || row.preview_note,
  })));
  byId("run-records").textContent = pretty(run.preview);
  byId("resume").disabled = run.active || run.status === "needs_reconciliation";
  clearTimeout(state.timer);
  if (run.active) state.timer = setTimeout(() => {
    Promise.all([showRun(id), refreshRuns()]).catch(error => message(error.message, true));
  }, 2000);
}
async function launch(resume = false) {
  if (!resume && state.inspected !== byId("workflow").value) throw new Error("Configuration changed; check prerequisites again.");
  const payload = {allow_network: byId("network").checked};
  if (resume) payload.id = state.selected.id; else payload.workflow = object("workflow");
  byId("launch").disabled = true;
  const run = await api(resume ? "resume" : "launch", payload);
  await refreshRuns(); await showRun(run.id);
  message("Workflow submitted. Progress and terminal results appear under Saved runs.");
  byId("history").scrollIntoView({behavior: "auto"});
}
async function init() {
  state.catalog = await api("catalog");
  byId("workspace").textContent = state.catalog.workspace || "Configuration builder only";
  byId("workspace-notice").hidden = state.catalog.execution_enabled;
  const option = (id, value, label) => {const node = document.createElement("option"); node.value=value; node.textContent=label; byId(id).append(node);};
  state.catalog.presets.forEach(p => option("preset", p.id, p.label));
  state.catalog.recipes.forEach(r => option("method", r.release_id, `${r.release_id} · ${r.route}`));
  state.catalog.benchmarks.forEach(b => option("benchmark", b.id, b.label));
  applyPreset();
  byId("builder").addEventListener("submit", event => {event.preventDefault(); task(generate);});
  byId("builder").addEventListener("input", () => invalidate("Form settings changed. Generate configuration to apply them; existing JSON is preserved."));
  byId("preset").addEventListener("change", applyPreset);
  byId("method").addEventListener("change", recipeNote);
  byId("benchmark").addEventListener("change", () => {byId("query").disabled = Boolean(byId("benchmark").value);});
  byId("workflow").addEventListener("input", () => {invalidate(); byId("inspect").disabled = !state.catalog.execution_enabled; byId("export-workflow").disabled = false; byId("export-run").disabled = false;});
  byId("probe").addEventListener("change", () => invalidate());
  byId("inspect").addEventListener("click", () => task(inspect));
  byId("launch").addEventListener("click", () => task(() => launch()));
  byId("resume").addEventListener("click", () => task(() => launch(true)));
  byId("refresh").addEventListener("click", () => task(refreshRuns));
  byId("load-run").addEventListener("click", () => {setDraft(state.selected.workflow); message("Saved configuration loaded. Editing creates a new run; Resume always uses the original.");});
  byId("export-workflow").addEventListener("click", () => task(async () => {download(object("workflow"), "workflow.json"); message("workflow.json exported.");}));
  byId("export-run").addEventListener("click", () => task(async () => {const config = object("workflow"); if (!config.run) throw new Error("Workflow requires a run object"); download(config.run, "run.json"); message("run.json exported.");}));
  byId("import").addEventListener("change", event => task(async () => {
    const file = event.target.files[0]; if (!file) return;
    if (file.size > 60000) throw new Error("Use a workflow smaller than 60 KB; reference large artifacts by path.");
    let value = JSON.parse(await file.text());
    if (value.release_id && !value.run) {
      const settings = {run: value, query: byId("query").value, limit: Number(byId("limit").value)};
      if (byId("benchmark").value) settings.benchmark_id = byId("benchmark").value;
      const generated = await api("template", settings); value = generated.workflow;
      byId("parameter-reference").textContent = pretty(generated.parameters);
    }
    if (!value.run || !value.target) throw new Error("Import a workflow.json with run/target, or a run.json with release_id.");
    setDraft(value); message("Workflow imported. Relative paths resolve against the server workspace. Check prerequisites before running.");
  }));
  await generate(); await refreshRuns();
  const match = location.hash.match(/^#run=([0-9a-f]{32})$/); if (match) await showRun(match[1]);
}
init().catch(error => message(error.message, true));
