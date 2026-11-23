"use strict";

const state = {
  catalog: null,
  validation: null,
  validationSequence: 0,
  validationTimer: null,
  toastTimer: null,
  activeRuns: 0,
};

const el = (id) => document.getElementById(id);

async function requestJSON(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json" } : undefined,
  });
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(payload?.error?.message || `Request failed (${response.status})`);
    error.status = response.status;
    error.payload = payload;
    throw error;
  }
  return payload;
}

function addOption(select, value, label) {
  const option = document.createElement("option");
  option.value = value;
  option.textContent = label;
  select.append(option);
}

function currentAlgorithm() {
  return state.catalog?.algorithms.find((item) => item.id === el("algorithm").value) || null;
}

function currentRelease() {
  return currentAlgorithm()?.versions.find((item) => item.id === el("release").value) || null;
}

function currentDataset() {
  return state.catalog?.datasets.find((item) => item.id === el("dataset").value) || null;
}

function currentBackend() {
  return state.catalog?.backends.find((item) => item.id === el("backend").value) || null;
}

function setSelectValue(select, desired, fallback = "") {
  const exists = Array.from(select.options).some((option) => option.value === desired);
  select.value = exists ? desired : fallback || select.options[0]?.value || "";
}

function populateCatalog() {
  const defaults = state.catalog.default_config;
  const algorithmSelect = el("algorithm");
  const backendSelect = el("backend");
  const datasetSelect = el("dataset");
  algorithmSelect.replaceChildren();
  backendSelect.replaceChildren();
  datasetSelect.replaceChildren();

  for (const algorithm of state.catalog.algorithms) {
    const readyCount = algorithm.versions.filter((version) => version.studio_ready).length;
    const suffix = readyCount ? `${readyCount} local` : "inspect only";
    addOption(algorithmSelect, algorithm.id, `${algorithm.name} · ${suffix}`);
  }
  for (const backend of state.catalog.backends) {
    addOption(backendSelect, backend.id, `${backend.name} · ${backend.ready ? "ready" : "locked"}`);
  }
  for (const dataset of state.catalog.datasets) {
    const readiness = dataset.execution_ready
      ? "executable"
      : dataset.data_ready
        ? "data ready"
        : "registered";
    addOption(datasetSelect, dataset.id, `${dataset.suite} / ${dataset.task} · ${readiness}`);
  }

  setSelectValue(algorithmSelect, defaults.algorithm_id);
  populateReleases(defaults.release_id);
  setSelectValue(backendSelect, defaults.backend_id);
  setSelectValue(datasetSelect, defaults.dataset_id);
  el("seed").value = defaults.parameters.seed;
  el("sample-limit").value = defaults.parameters.sample_limit;
  el("top-k").value = defaults.parameters.top_k;
  el("temperature").value = defaults.parameters.temperature;
  el("soft-word-budget").value = defaults.parameters.soft_word_budget;
  el("clarification-threshold").value = defaults.parameters.clarification_threshold;
  el("paired-baseline").checked = defaults.fairness.paired_baseline;
  el("fixed-seed").checked = defaults.fairness.fixed_seed;
  el("shared-retriever").checked = defaults.fairness.shared_retriever;
  el("freeze-dependencies").checked = defaults.fairness.freeze_dependencies;
  el("fail-on-drift").checked = defaults.fairness.fail_on_fingerprint_drift;
  el("catalog-count").textContent = `${state.catalog.algorithms.length - 1} methods · ${state.catalog.selectors.length} selectors · ${state.catalog.datasets.length - 1} benchmark cells`;
  updateContextCopy();
}

function populateReleases(preferred = "") {
  const select = el("release");
  select.replaceChildren();
  const algorithm = currentAlgorithm();
  for (const version of algorithm?.versions || []) {
    const availability = version.studio_ready ? "LOCAL" : "LOCKED";
    addOption(select, version.id, `${version.id} · ${availability}`);
  }
  setSelectValue(select, preferred);
  updateContextCopy();
}

function updateContextCopy() {
  const algorithm = currentAlgorithm();
  const release = currentRelease();
  const dataset = currentDataset();
  const backend = currentBackend();
  if (algorithm) {
    el("algorithm-family").textContent = `${algorithm.family} · ${algorithm.tier} · ${algorithm.status}`;
  }
  if (release) {
    el("release-status").textContent = `${release.status} · ${release.delivery_tier}`;
    el("method-note").querySelector("p").textContent = release.studio_reason;
  }
  if (backend) {
    el("backend-note").textContent = backend.description;
  }
  if (dataset) {
    const readiness = dataset.execution_ready
      ? "Data verified · executable"
      : dataset.data_ready
        ? "Data verified · execution locked"
        : "Data not prepared · execution locked";
    el("dataset-readiness").textContent = `${readiness} · ${dataset.reason}`;
    el("dataset-task").textContent = dataset.task;
    el("dataset-size").textContent = dataset.size;
    el("dataset-metrics").textContent = dataset.metrics.join(", ");
    el("dataset-families").textContent = dataset.families.join(", ");
    el("dataset-sample").textContent = dataset.sample
      ? `${dataset.sample.id}: “${dataset.sample.query}” → ${dataset.sample.target_document}`
      : "No local sample is exposed until the prepared-data adapter passes validation.";
  }
  el("manifest-release").textContent = el("release").value || "—";
  el("manifest-backend").textContent = el("backend").value || "—";
  el("manifest-dataset").textContent = el("dataset").value || "—";
}

function numericValue(id) {
  const value = el(id).valueAsNumber;
  return Number.isFinite(value) ? value : null;
}

function collectConfig() {
  return {
    algorithm_id: el("algorithm").value,
    release_id: el("release").value,
    backend_id: el("backend").value,
    dataset_id: el("dataset").value,
    parameters: {
      seed: numericValue("seed"),
      sample_limit: numericValue("sample-limit"),
      top_k: numericValue("top-k"),
      temperature: numericValue("temperature"),
      soft_word_budget: numericValue("soft-word-budget"),
      clarification_threshold: numericValue("clarification-threshold"),
    },
    fairness: {
      paired_baseline: el("paired-baseline").checked,
      fixed_seed: el("fixed-seed").checked,
      shared_retriever: el("shared-retriever").checked,
      freeze_dependencies: el("freeze-dependencies").checked,
      fail_on_fingerprint_drift: el("fail-on-drift").checked,
    },
  };
}

function scheduleValidation() {
  updateContextCopy();
  clearTimeout(state.validationTimer);
  el("validation-state").className = "validation-state checking";
  el("validation-state").textContent = "CHECKING";
  el("run-button").disabled = true;
  const sequence = ++state.validationSequence;
  state.validationTimer = setTimeout(() => validateCurrent(sequence), 120);
}

async function validateCurrent(sequence) {
  try {
    const report = await requestJSON("/api/validate", {
      method: "POST",
      body: JSON.stringify(collectConfig()),
    });
    if (sequence !== state.validationSequence) return;
    state.validation = report;
    renderValidation(report);
  } catch (error) {
    if (sequence !== state.validationSequence) return;
    renderValidation({
      valid: false,
      errors: [{ path: "api", message: error.message }],
      warnings: [],
      compatibility: { confirmatory: false },
    });
  }
}

function renderValidation(report) {
  const badge = el("validation-state");
  const list = el("issue-list");
  list.replaceChildren();
  badge.className = `validation-state ${report.valid ? "valid" : "invalid"}`;
  badge.textContent = report.valid ? "READY" : "BLOCKED";
  el("run-button").disabled = !report.valid;
  const allIssues = [...(report.errors || []), ...(report.warnings || [])];
  if (report.valid && !allIssues.length) {
    el("compatibility-title").textContent = "Ready for a controlled run";
    el("compatibility-copy").textContent = "Profile, replay backend, fixture data, and all fairness controls are compatible.";
    const item = document.createElement("li");
    item.className = "ok";
    item.textContent = "No compatibility exceptions";
    list.append(item);
  } else if (report.valid) {
    el("compatibility-title").textContent = "Ready, with exploratory flags";
    el("compatibility-copy").textContent = "The run can execute, but disabled controls will be recorded in its manifest.";
  } else {
    el("compatibility-title").textContent = "Configuration is blocked";
    el("compatibility-copy").textContent = "Resolve the constraints below before this experiment can enter the queue.";
  }
  for (const issue of report.errors || []) {
    const item = document.createElement("li");
    item.textContent = `${issue.path}: ${issue.message}`;
    list.append(item);
  }
  for (const issue of report.warnings || []) {
    const item = document.createElement("li");
    item.className = "warning";
    item.textContent = `${issue.path}: ${issue.message}`;
    list.append(item);
  }
  el("manifest-mode").textContent = report.compatibility?.confirmatory ? "confirmatory" : "exploratory";
}

function setupSteps() {
  for (const button of document.querySelectorAll(".step")) {
    button.addEventListener("click", () => {
      const selected = button.dataset.step;
      for (const step of document.querySelectorAll(".step")) {
        const active = step === button;
        step.classList.toggle("active", active);
        step.setAttribute("aria-selected", String(active));
      }
      for (const panel of document.querySelectorAll(".config-section")) {
        const active = panel.dataset.panel === selected;
        panel.hidden = !active;
        panel.classList.toggle("active", active);
      }
    });
  }
}

function bindInputs() {
  el("algorithm").addEventListener("change", () => {
    populateReleases();
    scheduleValidation();
  });
  for (const input of document.querySelectorAll("#run-form select:not(#algorithm), #run-form input")) {
    input.addEventListener("input", scheduleValidation);
    input.addEventListener("change", scheduleValidation);
  }
  el("run-form").addEventListener("submit", queueRun);
  el("refresh-runs").addEventListener("click", refreshRuns);
}

async function queueRun(event) {
  event.preventDefault();
  if (!state.validation?.valid) return;
  const button = el("run-button");
  button.classList.add("running");
  button.querySelector("span").textContent = "Queueing";
  button.disabled = true;
  try {
    const run = await requestJSON("/api/runs", {
      method: "POST",
      body: JSON.stringify(collectConfig()),
    });
    showToast(`Run ${run.id.slice(0, 8)} entered the local queue.`);
    await refreshRuns();
    document.getElementById("runs").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    const details = error.payload?.errors?.[0]?.message;
    showToast(details || error.message);
  } finally {
    button.classList.remove("running");
    button.querySelector("span").textContent = "Queue experiment";
    button.disabled = !state.validation?.valid;
  }
}

function short(value, size = 30) {
  const text = String(value ?? "—");
  return text.length > size ? `${text.slice(0, size - 1)}…` : text;
}

function formatPercent(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : "—";
}

function tableCell(row, text, className = "") {
  const cell = document.createElement("td");
  cell.textContent = text;
  if (className) cell.className = className;
  row.append(cell);
  return cell;
}

function renderRuns(runs) {
  const body = el("runs-body");
  body.replaceChildren();
  if (!runs.length) {
    const row = document.createElement("tr");
    const cell = tableCell(row, "No experiments queued in this session.", "empty-row");
    cell.colSpan = 7;
    body.append(row);
    el("queue-summary").textContent = "No active runs";
    return;
  }
  let active = 0;
  for (const run of runs) {
    if (["queued", "running"].includes(run.status)) active += 1;
    const row = document.createElement("tr");
    tableCell(row, run.id.slice(0, 8));
    tableCell(row, short(run.config.release_id, 28));
    tableCell(row, short(run.config.dataset_id, 24));
    const statusCell = document.createElement("td");
    const status = document.createElement("span");
    status.className = `run-table-status ${run.status}`;
    status.textContent = run.status.replaceAll("_", " ");
    statusCell.append(status);
    row.append(statusCell);
    const quality = run.result?.summary?.quality?.treatment;
    tableCell(row, formatPercent(quality));
    const cost = run.result?.summary?.usage?.cost_usd;
    tableCell(row, Number.isFinite(cost) ? `$${cost.toFixed(4)}` : "—");
    const linkCell = document.createElement("td");
    const link = document.createElement("a");
    link.className = "result-link";
    link.href = `/result?run=${encodeURIComponent(run.id)}`;
    link.textContent = "Open →";
    linkCell.append(link);
    row.append(linkCell);
    body.append(row);
  }
  state.activeRuns = active;
  el("queue-summary").textContent = active ? `${active} active · ${runs.length} recorded` : `${runs.length} recorded · queue idle`;
}

async function refreshRuns() {
  try {
    const payload = await requestJSON("/api/runs");
    renderRuns(payload.runs);
  } catch (error) {
    showToast(`Run queue unavailable: ${error.message}`);
  }
}

function showToast(message) {
  const toast = el("toast");
  toast.textContent = message;
  toast.classList.add("visible");
  clearTimeout(state.toastTimer);
  state.toastTimer = setTimeout(() => toast.classList.remove("visible"), 3600);
}

async function init() {
  setupSteps();
  try {
    const [catalog] = await Promise.all([
      requestJSON("/api/catalog"),
      requestJSON("/api/health"),
    ]);
    state.catalog = catalog;
    populateCatalog();
    bindInputs();
    scheduleValidation();
    await refreshRuns();
  } catch (error) {
    el("api-dot").classList.add("offline");
    el("catalog-count").textContent = "Studio API unavailable";
    showToast(error.message);
  }
  setInterval(() => {
    if (!document.hidden && state.activeRuns > 0) refreshRuns();
  }, 1500);
}

document.addEventListener("DOMContentLoaded", init);
