"use strict";

const el = (id) => document.getElementById(id);
let activeRecord = null;
let pollTimer = null;

async function requestJSON(url) {
  const response = await fetch(url);
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload?.error?.message || `Request failed (${response.status})`);
  return payload;
}

function formatPercent(value, signed = false) {
  if (!Number.isFinite(value)) return "—";
  const prefix = signed && value > 0 ? "+" : "";
  return `${prefix}${(value * 100).toFixed(1)}%`;
}

function formatNumber(value, digits = 2) {
  return Number.isFinite(value) ? value.toFixed(digits) : "—";
}

function formatStatus(status) {
  return String(status || "waiting").replaceAll("_", " ").toUpperCase();
}

function backendAuditLabel(backend) {
  const backendId = typeof backend?.backend_id === "string"
    ? backend.backend_id
    : "backend identity unavailable";
  const fingerprint = backend?.actual_backend_fingerprint_sha256
    || backend?.model_id_sha256;
  return typeof fingerprint === "string" && /^[0-9a-f]{64}$/.test(fingerprint)
    ? `${backendId} · ${fingerprint.slice(0, 12)}`
    : backendId;
}

function setStatus(status) {
  const badge = el("run-status");
  badge.className = `run-status ${status || "waiting"}`;
  badge.textContent = formatStatus(status);
}

function setDefinitionList(target, entries) {
  target.replaceChildren();
  for (const [label, value] of entries) {
    const row = document.createElement("div");
    const term = document.createElement("dt");
    const description = document.createElement("dd");
    term.textContent = label;
    description.textContent = value;
    row.append(term, description);
    target.append(row);
  }
}

function createText(tag, text, className = "") {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
}

function renderSamples(samples) {
  const list = el("sample-list");
  list.replaceChildren();
  for (const [index, sample] of samples.entries()) {
    const details = document.createElement("details");
    details.className = "sample-item";
    const summary = document.createElement("summary");
    summary.append(
      createText("span", String(index + 1).padStart(2, "0"), "sample-index"),
      createText("span", sample.original_query, "sample-query"),
      createText("span", `B ${formatPercent(sample.baseline_score)}`, "sample-score"),
      createText("span", `T ${formatPercent(sample.treatment_score)}`, "sample-score"),
      createText("span", "+", "sample-chevron"),
    );
    const detail = document.createElement("div");
    detail.className = "sample-detail";
    const before = document.createElement("div");
    before.append(createText("h3", "ORIGINAL STATE"), createText("p", sample.original_query));
    const after = document.createElement("div");
    after.append(
      createText("h3", "TRANSFORMED STATE"),
      createText("p", sample.transformed_query),
    );
    if (sample.transformed_instruction) {
      after.append(createText("h3", "INSTRUCTION"), createText("p", sample.transformed_instruction));
    }
    const metadata = document.createElement("dl");
    setDefinitionList(metadata, [
      ["Target", sample.target_document_id],
      ["Baseline top", sample.baseline_top_document_id || "none"],
      ["Treatment top", sample.treatment_top_document_id || "none"],
      ["Latency", `${formatNumber(sample.latency_ms, 3)} ms`],
      ["Patch", sample.patch_hash.slice(0, 12)],
    ]);
    after.append(metadata);
    detail.append(before, after);
    details.append(summary, detail);
    list.append(details);
  }
}

function renderFailures(failures) {
  const list = el("failure-list");
  list.replaceChildren();
  el("failure-summary").textContent = failures.length ? `${failures.length} failures recorded` : "No failures recorded";
  if (!failures.length) {
    list.append(createText("p", "All requested examples reached a terminal result.", "no-failures"));
    return;
  }
  for (const failure of failures) {
    const row = document.createElement("div");
    row.className = "failure-item";
    row.append(
      createText("strong", `${failure.example_id} · ${failure.category}`),
      createText("span", failure.message),
      createText("span", `${formatNumber(failure.latency_ms, 2)} ms`),
    );
    list.append(row);
  }
}

function summarizePayload(payload) {
  const rendered = JSON.stringify(payload);
  return rendered.length > 420 ? `${rendered.slice(0, 419)}…` : rendered;
}

function renderTrace(samples) {
  const list = el("trace-list");
  list.replaceChildren();
  for (const sample of samples) {
    const trace = sample.trace;
    const group = document.createElement("div");
    group.className = "trace-group";
    const identity = document.createElement("div");
    identity.append(
      createText("strong", sample.example_id),
      createText("small", trace.release_id),
      createText("small", sample.patch_hash),
    );
    const events = document.createElement("div");
    events.className = "trace-events";
    for (const event of trace.events || []) {
      const item = document.createElement("div");
      item.className = "trace-event";
      item.append(
        createText("strong", `${event.phase} · ${event.name}`),
        createText("p", summarizePayload(event.payload)),
      );
      events.append(item);
    }
    for (const event of sample.observability?.snapshot?.events || []) {
      const item = document.createElement("div");
      item.className = "trace-event monitor-event";
      item.append(
        createText("strong", `monitor · ${event.kind} · ${event.component || "run"}`),
        createText("p", `${formatNumber(event.elapsed_ms, 3)} ms · ${summarizePayload(event.payload)}`),
      );
      events.append(item);
    }
    if (!(trace.events || []).length) {
      events.append(createText("p", "Typed NO_OP trace contains no method-specific events."));
    }
    group.append(identity, events);
    list.append(group);
  }
}

function renderResult(record) {
  activeRecord = record;
  setStatus(record.status);
  el("result-title").textContent = record.config.release_id;
  el("result-subtitle").textContent = `${record.config.dataset_id} · run ${record.id.slice(0, 12)} · created ${record.created_at}`;
  if (!record.result) {
    el("result-empty").hidden = false;
    el("result-empty").querySelector("h2").textContent = record.status === "failed" ? "Run failed before producing a result" : "Run is in progress";
    el("result-empty").querySelector("p").textContent = record.error?.message || "The result page will update automatically when the local worker finishes.";
    return;
  }
  el("result-empty").hidden = true;
  el("result-content").hidden = false;
  const summary = record.result.summary;
  const quality = summary.quality;
  const usage = summary.usage;
  const timing = summary.timing;
  el("metric-quality").textContent = formatPercent(quality.treatment);
  el("metric-quality-delta").textContent = `${formatPercent(quality.mean_effect, true)} vs baseline`;
  el("metric-cost").textContent = `$${formatNumber(usage.cost_usd, 4)}`;
  el("metric-tokens").textContent = String(usage.output_tokens_estimate ?? "—");
  el("metric-latency").textContent = `${formatNumber(timing.wall_ms, 1)} ms`;
  el("metric-failures").textContent = String(summary.failure_count);
  el("scope-label").textContent = record.result.result_scope.replaceAll("_", " ");
  el("baseline-value").textContent = formatPercent(quality.baseline);
  el("treatment-value").textContent = formatPercent(quality.treatment);
  el("effect-value").textContent = formatPercent(quality.mean_effect, true);
  el("rates-value").textContent = `${formatPercent(quality.win_rate)} / ${formatPercent(quality.tie_rate)} / ${formatPercent(quality.harm_rate)}`;
  el("intent-value").textContent = formatPercent(quality.intent_token_recall);
  el("jaccard-value").textContent = formatPercent(quality.token_jaccard);
  requestAnimationFrame(() => {
    el("baseline-bar").style.width = `${Math.max(0, Math.min(100, quality.baseline * 100))}%`;
    el("treatment-bar").style.width = `${Math.max(0, Math.min(100, quality.treatment * 100))}%`;
  });

  setDefinitionList(el("usage-list"), [
    ["Model calls", String(usage.model_calls)],
    ["Optimizer retrieval", String(usage.optimizer_retrieval_calls)],
    ["Evaluation retrieval", String(usage.evaluation_retrieval_calls)],
    ["Tool / embedding", `${usage.tool_calls} / ${usage.embedding_calls}`],
    ["Core tokens", `${usage.core_input_tokens} in / ${usage.core_output_tokens} out`],
    ["Input estimate", String(usage.input_tokens_estimate)],
    ["Output estimate", String(usage.output_tokens_estimate)],
    ["CPU", `${formatNumber(usage.cpu_seconds, 6)} s`],
    ["Cost basis", usage.cost_basis],
  ]);
  setDefinitionList(el("timing-list"), [
    ["Wall", `${formatNumber(timing.wall_ms, 3)} ms`],
    ["Mean sample", `${formatNumber(timing.mean_sample_ms, 3)} ms`],
    ["Median sample", `${formatNumber(timing.median_sample_ms, 3)} ms`],
    ["P95 sample", `${formatNumber(timing.p95_sample_ms, 3)} ms`],
    ["Core intervention", `${formatNumber(timing.core_intervention_ms, 3)} ms`],
    ["Core mean", `${formatNumber(timing.core_mean_intervention_ms, 3)} ms`],
    ["Started", record.started_at || "—"],
    ["Finished", record.finished_at || "—"],
  ]);
  const audit = record.result.fairness_audit;
  setDefinitionList(el("fairness-list"), [
    ["Seed", String(audit.seed)],
    ["Fingerprint drift", String(audit.fingerprint_drift)],
    ["Retriever", audit.retriever_before.retriever_id],
    ["Corpus", audit.retriever_before.corpus_hash.slice(0, 12)],
    ["Backend", backendAuditLabel(audit.backend)],
    ["Dataset revision", audit.dataset_revision],
  ]);
  renderSamples(record.result.samples);
  renderFailures(record.result.failures);
  renderTrace(record.result.samples);
  el("raw-json").textContent = JSON.stringify(record, null, 2);
}

async function loadRun(runId) {
  try {
    const record = await requestJSON(`/api/runs/${encodeURIComponent(runId)}`);
    renderResult(record);
    if (["queued", "running"].includes(record.status)) {
      clearTimeout(pollTimer);
      pollTimer = setTimeout(() => loadRun(runId), 750);
    }
  } catch (error) {
    setStatus("failed");
    el("result-empty").hidden = false;
    el("result-empty").querySelector("h2").textContent = "Result is unavailable";
    el("result-empty").querySelector("p").textContent = error.message;
  }
}

function init() {
  const runId = new URLSearchParams(window.location.search).get("run");
  el("toggle-json").addEventListener("click", () => {
    const raw = el("raw-json");
    raw.hidden = !raw.hidden;
    el("toggle-json").textContent = raw.hidden ? "Raw JSON" : "Hide JSON";
  });
  if (!runId) return;
  el("result-empty").hidden = false;
  el("result-empty").querySelector("h2").textContent = "Loading run";
  el("result-empty").querySelector("p").textContent = "Reading the local run record and measurements.";
  loadRun(runId);
}

document.addEventListener("DOMContentLoaded", init);
