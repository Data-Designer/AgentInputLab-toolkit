const campaignState = {
  catalog: null,
  profile: null,
  preparation: null,
  executions: [],
  pollTimer: null,
};

const byId = (id) => document.getElementById(id);

function shortHash(value, width = 12) {
  return typeof value === "string" && value.length > width
    ? `${value.slice(0, width)}…`
    : value || "—";
}

function showCampaignToast(message) {
  const toast = byId("campaign-toast");
  toast.textContent = message;
  toast.classList.add("visible");
  window.setTimeout(() => toast.classList.remove("visible"), 2600);
}

async function campaignRequest(path, options = {}) {
  const response = await fetch(path, {
    headers: options.body ? { "Content-Type": "application/json" } : {},
    ...options,
  });
  const payload = await response.json();
  if (!response.ok) {
    const message = payload.error && payload.error.message
      ? payload.error.message
      : `Request failed (${response.status})`;
    throw new Error(message);
  }
  return payload;
}

function profileButton(profile) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "campaign-profile-option";
  button.setAttribute("role", "option");
  button.setAttribute(
    "aria-selected",
    String(campaignState.profile && campaignState.profile.id === profile.id),
  );

  const index = document.createElement("span");
  index.textContent = String(
    campaignState.catalog.profiles.indexOf(profile) + 1,
  ).padStart(2, "0");
  const copy = document.createElement("span");
  const name = document.createElement("strong");
  name.textContent = profile.label;
  const mode = document.createElement("small");
  mode.textContent = profile.execution_configured
    ? "Preflight + confirmed execution"
    : "Preflight / plan only";
  copy.append(name, mode);
  const arrow = document.createElement("b");
  arrow.setAttribute("aria-hidden", "true");
  arrow.textContent = "→";
  button.append(index, copy, arrow);
  button.addEventListener("click", () => selectCampaignProfile(profile));
  return button;
}

function renderCampaignProfiles() {
  const profiles = campaignState.catalog ? campaignState.catalog.profiles : [];
  byId("campaign-profile-list").replaceChildren(
    ...profiles.map(profileButton),
  );
}

function resetCampaignPreparation() {
  campaignState.preparation = null;
  byId("campaign-preflight-state").textContent = "NOT RUN";
  byId("campaign-preflight-state").className = "validation-state waiting";
  byId("campaign-preflight-title").textContent = "No materialized plan";
  byId("campaign-preflight-copy").textContent =
    "Preflight checks assets and drivers without executing a benchmark cell.";
  byId("campaign-stat-cells").textContent = "—";
  byId("campaign-stat-targets").textContent = "—";
  byId("campaign-stat-arms").textContent = "—";
  byId("campaign-stat-grid").textContent = "—";
  byId("campaign-blocker-count").textContent = "0 blockers";
  const initial = document.createElement("li");
  initial.className = "ok";
  initial.textContent = "Materialize this profile to inspect readiness.";
  byId("campaign-blockers").replaceChildren(initial);
  const placeholder = document.createElement("p");
  placeholder.className = "campaign-placeholder";
  placeholder.textContent = "Arm identities appear after preflight.";
  byId("campaign-arms").replaceChildren(placeholder);
  byId("campaign-study-stage").textContent = "—";
  byId("campaign-plan-identity").textContent = "No plan hash";
  const row = document.createElement("tr");
  const cell = document.createElement("td");
  cell.colSpan = 6;
  cell.className = "empty-row";
  cell.textContent = "Preflight has not materialized cells.";
  row.append(cell);
  byId("campaign-cells-body").replaceChildren(row);
  byId("campaign-confirm-checkbox").checked = false;
  byId("campaign-confirm-checkbox").disabled = true;
  byId("campaign-execute-button").disabled = true;
  byId("campaign-execution-note").textContent =
    "Execution stays locked until the profile, assets, output root, and plan are ready.";
}

function selectCampaignProfile(profile) {
  campaignState.profile = profile;
  renderCampaignProfiles();
  resetCampaignPreparation();
  byId("campaign-profile-name").textContent = profile.label;
  byId("campaign-profile-description").textContent = profile.description;
  byId("campaign-profile-identity").textContent = shortHash(
    profile.profile_identity_sha256,
    18,
  );
  byId("campaign-blueprint-identity").textContent = shortHash(
    profile.blueprint_sha256,
    18,
  );
  byId("campaign-arm-catalog-identity").textContent = shortHash(
    profile.arm_catalog_sha256,
    18,
  );
  byId("campaign-preflight-button").disabled = false;
}

function issueNode(message, kind) {
  const item = document.createElement("li");
  item.className = kind;
  item.textContent = message;
  return item;
}

function armNode(arm) {
  const row = document.createElement("div");
  row.className = "campaign-arm-row";
  const identity = document.createElement("span");
  const armId = document.createElement("strong");
  armId.textContent = arm.arm_id;
  const release = document.createElement("small");
  release.textContent = arm.release_id;
  identity.append(armId, release);
  const family = document.createElement("span");
  family.className = "mono-tag";
  family.textContent = arm.family;
  row.append(identity, family);
  return row;
}

function cellRow(cell) {
  const row = document.createElement("tr");
  const values = [
    shortHash(cell.cell_id, 9),
    `${cell.benchmark_id} / ${cell.split}`,
    cell.unit_id,
    cell.model_slot,
    String(cell.seed),
    cell.arm_ids.join(", "),
  ];
  values.forEach((value) => {
    const column = document.createElement("td");
    column.textContent = value;
    row.append(column);
  });
  return row;
}

function renderCampaignPreparation(preparation) {
  campaignState.preparation = preparation;
  const ready = preparation.preflight.ready;
  const plan = preparation.plan;
  byId("campaign-preflight-state").textContent = ready ? "READY" : "BLOCKED";
  byId("campaign-preflight-state").className = ready
    ? "validation-state valid"
    : "validation-state invalid";
  byId("campaign-preflight-title").textContent = ready
    ? "Exact plan materialized"
    : "Preflight found blockers";
  byId("campaign-preflight-copy").textContent = ready
    ? "No benchmark cell has run. Review every identity before confirmation."
    : "Resolve the server-side prerequisites, then preflight this profile again.";
  byId("campaign-stat-cells").textContent = plan ? String(plan.cell_count) : "0";
  byId("campaign-stat-targets").textContent = String(
    preparation.manifest.benchmarks.length,
  );
  byId("campaign-stat-arms").textContent = String(
    preparation.manifest.arms.length,
  );
  byId("campaign-stat-grid").textContent =
    `${preparation.manifest.model_slots.length} × ${preparation.manifest.seeds.length}`;
  byId("campaign-study-stage").textContent =
    `${preparation.manifest.stage} · ${preparation.manifest.estimand_profile_kind}`;

  const blockers = preparation.preflight.blockers;
  const warnings = preparation.preflight.warnings;
  byId("campaign-blocker-count").textContent =
    `${blockers.length} blocker${blockers.length === 1 ? "" : "s"}`;
  const issues = [];
  blockers.forEach((message) => issues.push(issueNode(message, "")));
  warnings.forEach((message) => issues.push(issueNode(message, "warning")));
  if (!issues.length) {
    issues.push(issueNode("All configured preflight gates passed.", "ok"));
  }
  byId("campaign-blockers").replaceChildren(...issues);
  byId("campaign-arms").replaceChildren(
    ...preparation.manifest.arms.map(armNode),
  );

  if (plan) {
    byId("campaign-plan-identity").textContent =
      `PLAN ${shortHash(plan.plan_hash, 18)}`;
    const rows = plan.cells.map(cellRow);
    if (plan.preview_truncated) {
      const truncated = document.createElement("tr");
      const note = document.createElement("td");
      note.colSpan = 6;
      note.className = "empty-row";
      note.textContent =
        `Showing ${plan.preview_count} of ${plan.cell_count} frozen cells.`;
      truncated.append(note);
      rows.push(truncated);
    }
    byId("campaign-cells-body").replaceChildren(...rows);
  } else {
    byId("campaign-plan-identity").textContent = "No plan hash";
    const row = document.createElement("tr");
    const note = document.createElement("td");
    note.colSpan = 6;
    note.className = "empty-row";
    note.textContent = "Blocked preflight does not create an executable plan.";
    row.append(note);
    byId("campaign-cells-body").replaceChildren(row);
  }

  const canConfirm = preparation.execution.ready_for_confirmation;
  byId("campaign-confirm-checkbox").checked = false;
  byId("campaign-confirm-checkbox").disabled = !canConfirm;
  byId("campaign-execute-button").disabled = true;
  if (canConfirm) {
    byId("campaign-execution-note").textContent =
      "This profile is execution-enabled. Confirmation will queue only the exact displayed plan hash.";
  } else if (!preparation.profile.execution_enabled) {
    byId("campaign-execution-note").textContent =
      "The server profile is intentionally preflight-only.";
  } else if (!preparation.execution.output_configured) {
    byId("campaign-execution-note").textContent =
      "No server-side campaign output root is configured.";
  } else if (preparation.execution.blockers.length) {
    byId("campaign-execution-note").textContent =
      `Execution safety blocker: ${preparation.execution.blockers.join("; ")}`;
  } else {
    byId("campaign-execution-note").textContent =
      "Execution remains locked while preflight has blockers.";
  }
}

async function preflightCampaign() {
  if (!campaignState.profile) return;
  const button = byId("campaign-preflight-button");
  button.disabled = true;
  button.classList.add("running");
  byId("campaign-preflight-state").textContent = "CHECKING";
  byId("campaign-preflight-state").className = "validation-state checking";
  try {
    const preparation = await campaignRequest("/api/campaigns/preflight", {
      method: "POST",
      body: JSON.stringify({ profile_id: campaignState.profile.id }),
    });
    renderCampaignPreparation(preparation);
    showCampaignToast(
      preparation.preflight.ready
        ? "Exact plan materialized. No cells executed."
        : "Preflight completed with blockers.",
    );
  } catch (error) {
    resetCampaignPreparation();
    byId("campaign-preflight-state").textContent = "FAILED";
    byId("campaign-preflight-state").className = "validation-state invalid";
    showCampaignToast(error.message);
  } finally {
    button.classList.remove("running");
    button.disabled = false;
  }
}

function executionRow(record) {
  const row = document.createElement("tr");
  const progress = record.progress;
  const resultIdentity = record.result_identity
    ? shortHash(record.result_identity.artifact_identity_sha256, 12)
    : "—";
  const values = [
    shortHash(record.id, 9),
    record.profile_id,
    record.status,
    `${progress.completed_cells}/${progress.total_cells}`,
    shortHash(record.plan_hash, 10),
    resultIdentity,
  ];
  values.forEach((value, index) => {
    const cell = document.createElement("td");
    cell.textContent = value;
    if (index === 2) {
      const status = document.createElement("span");
      status.className = `run-table-status ${record.status}`;
      status.textContent = value;
      cell.replaceChildren(status);
    }
    row.append(cell);
  });
  return row;
}

function renderCampaignExecutions() {
  const body = byId("campaign-executions-body");
  if (!campaignState.executions.length) {
    const row = document.createElement("tr");
    const cell = document.createElement("td");
    cell.colSpan = 6;
    cell.className = "empty-row";
    cell.textContent = "No campaign execution has been queued.";
    row.append(cell);
    body.replaceChildren(row);
    return;
  }
  body.replaceChildren(...campaignState.executions.map(executionRow));
}

async function refreshCampaignExecutions() {
  try {
    const payload = await campaignRequest("/api/campaigns/executions");
    campaignState.executions = payload.executions;
    renderCampaignExecutions();
    const active = payload.executions.some((record) =>
      ["queued", "running"].includes(record.status)
    );
    if (active && !campaignState.pollTimer) {
      campaignState.pollTimer = window.setInterval(
        refreshCampaignExecutions,
        1600,
      );
    } else if (!active && campaignState.pollTimer) {
      window.clearInterval(campaignState.pollTimer);
      campaignState.pollTimer = null;
    }
  } catch (error) {
    showCampaignToast(error.message);
  }
}

async function executeCampaign() {
  const preparation = campaignState.preparation;
  if (!preparation || !preparation.plan) return;
  const button = byId("campaign-execute-button");
  button.disabled = true;
  button.classList.add("running");
  try {
    await campaignRequest("/api/campaigns/executions", {
      method: "POST",
      body: JSON.stringify({
        profile_id: preparation.profile.id,
        plan_id: preparation.plan.id,
        plan_hash: preparation.plan.plan_hash,
        confirmation: preparation.execution.confirmation_phrase,
      }),
    });
    byId("campaign-confirm-checkbox").checked = false;
    byId("campaign-confirm-checkbox").disabled = true;
    showCampaignToast("Confirmed frozen plan queued.");
    await refreshCampaignExecutions();
  } catch (error) {
    button.disabled = false;
    showCampaignToast(error.message);
  } finally {
    button.classList.remove("running");
  }
}

async function bootCampaigns() {
  try {
    const catalog = await campaignRequest("/api/campaigns/profiles");
    campaignState.catalog = catalog;
    byId("campaign-profile-count").textContent =
      `${catalog.profiles.length} allowlisted profile${catalog.profiles.length === 1 ? "" : "s"}`;
    const executable = catalog.profiles.some(
      (profile) => profile.execution_configured,
    );
    byId("campaign-mode-state").textContent = executable
      ? "CONFIRMED EXECUTION"
      : "PREFLIGHT ONLY";
    byId("campaign-unconfigured").hidden = catalog.configured;
    byId("campaign-profile-content").hidden = !catalog.configured;
    byId("campaign-api-dot").classList.toggle("offline", !catalog.configured);
    renderCampaignProfiles();
    if (catalog.profiles.length) selectCampaignProfile(catalog.profiles[0]);
    await refreshCampaignExecutions();
  } catch (error) {
    byId("campaign-unconfigured").hidden = false;
    byId("campaign-profile-content").hidden = true;
    byId("campaign-api-dot").classList.add("offline");
    showCampaignToast(error.message);
  }
}

byId("campaign-preflight-button").addEventListener("click", preflightCampaign);
byId("campaign-confirm-checkbox").addEventListener("change", (event) => {
  const preparation = campaignState.preparation;
  byId("campaign-execute-button").disabled = !(
    event.target.checked
    && preparation
    && preparation.execution.ready_for_confirmation
  );
});
byId("campaign-execute-button").addEventListener("click", executeCampaign);
byId("campaign-refresh-executions").addEventListener(
  "click",
  refreshCampaignExecutions,
);

bootCampaigns();
