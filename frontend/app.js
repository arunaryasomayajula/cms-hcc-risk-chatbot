/* ── State ──────────────────────────────────────────────────────────────────── */
let conversationHistory = [];
let systemReady = false;
let statusInterval = null;

/* ── DOM refs ───────────────────────────────────────────────────────────────── */
const messages    = document.getElementById('messages');
const msgInput    = document.getElementById('msgInput');
const sendBtn     = document.getElementById('sendBtn');
const clearBtn    = document.getElementById('clearBtn');
const sidebar     = document.getElementById('sidebar');
const sidebarToggle  = document.getElementById('sidebarToggle');
const sidebarOpenBtn = document.getElementById('sidebarOpenBtn');
const statusBadge = document.getElementById('statusBadge');
const extractProvider = document.getElementById('extractProvider');
const selectProvider  = document.getElementById('selectProvider');
const explainProvider = document.getElementById('explainProvider');
const extractModel    = document.getElementById('extractModel');
const selectModel     = document.getElementById('selectModel');
const explainModel    = document.getElementById('explainModel');
const providerNote    = document.getElementById('providerNote');
const hardwareNote    = document.getElementById('hardwareNote');

/* Fallback only — the server sends a `label` per provider. */
const PROVIDER_LABELS = {
  claude: 'Claude',
  medgemma: 'MedGemma (vLLM)',
  airllm: 'AirLLM (local)',
  ollama: 'Ollama (local)',
};

const STEP_SELECTS = {
  extract: { prov: extractProvider, model: extractModel },
  select:  { prov: selectProvider,  model: selectModel  },
  explain: { prov: explainProvider, model: explainModel },
};

let PROVIDER_CATALOG = [];
let MODEL_DEFAULTS = {};
let HARDWARE = null;

/* ── Sidebar toggle ─────────────────────────────────────────────────────────── */
sidebarToggle.addEventListener('click', () => {
  sidebar.classList.add('collapsed');
  sidebarOpenBtn.style.display = 'block';
});
sidebarOpenBtn.addEventListener('click', () => {
  sidebar.classList.remove('collapsed');
  sidebarOpenBtn.style.display = 'none';
});

/* ── Status polling ─────────────────────────────────────────────────────────── */
async function pollStatus() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();
    if (d.initialized) {
      systemReady = true;
      statusBadge.className = 'status-badge ready';
      statusBadge.textContent = '● Ready';
      sendBtn.disabled = false;
      clearInterval(statusInterval);
    } else if (d.error) {
      statusBadge.className = 'status-badge error';
      statusBadge.textContent = '● Error: ' + d.error.substring(0, 40);
    }
  } catch (_) {}
}
pollStatus();
statusInterval = setInterval(pollStatus, 5000);

/* ── Model provider config ──────────────────────────────────────────────────── */
function providerByName(name) {
  return PROVIDER_CATALOG.find(p => p.name === name);
}

function providerLabel(name) {
  return providerByName(name)?.label || PROVIDER_LABELS[name] || name;
}

/* Repopulate a step's model dropdown for whichever provider is selected. */
function renderModels(step) {
  const { prov, model } = STEP_SELECTS[step];
  if (!prov || !model) return;
  const p = providerByName(prov.value);
  if (!p) { model.innerHTML = ''; return; }

  const models = p.models || [];
  model.innerHTML = models.map(m =>
    `<option value="${escHtml(m.id)}" title="${escHtml(m.note || '')}">${escHtml(m.label || m.id)}</option>`
  ).join('');

  const wanted = MODEL_DEFAULTS[step];
  model.value = models.some(m => m.id === wanted)
    ? wanted
    : (p.default_model || models[0]?.id || '');
}

function anyLocalSelected() {
  return Object.values(STEP_SELECTS)
    .some(({ prov }) => providerByName(prov?.value)?.local);
}

function renderNotes(guard) {
  if (hardwareNote) {
    if (HARDWARE) {
      const warnings = (HARDWARE.warnings || [])
        .map(w => `<span class="hw-warning">⚠ ${escHtml(w)}</span>`).join('');
      hardwareNote.innerHTML =
        `<span class="hw-line">Local inference: ${escHtml(HARDWARE.summary || '')}</span>${warnings}`;
    } else {
      hardwareNote.innerHTML = '';
    }
  }

  if (!providerNote) return;
  const lines = PROVIDER_CATALOG.map(p => {
    if (!p.local) return '';
    const dot = p.reachable ? '🟢' : '⚪';
    const state = p.reachable ? escHtml(p.base_url) : 'not running';
    return `<span class="prov-line"><span class="prov-dot">${dot}</span> ${escHtml(p.label)} — ${state}</span>`;
  }).filter(Boolean).join('');

  const guardMsg = guard?.available
    ? 'PHI guardrail: on.'
    : `PHI guardrail: off (${escHtml(guard?.error || 'unavailable')}).`;
  providerNote.innerHTML = `${lines}<span class="prov-line">${guardMsg}</span>`;
}

async function loadConfig() {
  try {
    const r = await fetch('/api/config');
    const d = await r.json();

    PROVIDER_CATALOG = d.providers || [];
    MODEL_DEFAULTS = d.model_defaults || {};
    HARDWARE = d.hardware || null;
    const defaults = d.step_defaults || {};

    for (const [step, { prov }] of Object.entries(STEP_SELECTS)) {
      if (!prov) continue;
      prov.innerHTML = PROVIDER_CATALOG.map(p =>
        `<option value="${escHtml(p.name)}" title="${escHtml(p.note || '')}">${escHtml(p.label)}</option>`
      ).join('');
      prov.value = defaults[step] || PROVIDER_CATALOG[0]?.name || 'claude';
      renderModels(step);
      // Cascade: switching provider must rebuild that step's model list.
      prov.addEventListener('change', () => renderModels(step));
    }

    renderNotes(d.guardrails);
  } catch (_) {
    if (providerNote) providerNote.textContent = 'Could not load model config.';
  }
}
loadConfig();

function getProviders() {
  return {
    extract: extractProvider?.value || 'claude',
    select:  selectProvider?.value  || 'claude',
    explain: explainProvider?.value || 'claude',
  };
}

function getModels() {
  return {
    extract: extractModel?.value || '',
    select:  selectModel?.value  || '',
    explain: explainModel?.value || '',
  };
}

/* ── Demographics ───────────────────────────────────────────────────────────── */
function getDemographics() {
  return {
    dob:         document.getElementById('dob').value         || '1950-01-01',
    sex:     parseInt(document.getElementById('sex').value)   || 2,
    orec:    parseInt(document.getElementById('orec').value)  || 0,
    dual_status: parseInt(document.getElementById('dual_status').value) || 0,
    ltimcaid: document.getElementById('ltimcaid').checked ? 1 : 0,
    nemcaid:  document.getElementById('nemcaid').checked  ? 1 : 0,
  };
}

/* ── Message rendering ──────────────────────────────────────────────────────── */
function scrollBottom() {
  messages.scrollTop = messages.scrollHeight;
}

function addUserMessage(text) {
  const div = document.createElement('div');
  div.className = 'message user';
  div.innerHTML = `
    <div class="avatar">👤</div>
    <div class="bubble"><p>${escHtml(text)}</p></div>`;
  messages.appendChild(div);
  scrollBottom();
}

let localTimer = null;
let localPoll = null;
let activeController = null;

function addTyping() {
  const div = document.createElement('div');
  div.className = 'message bot';
  div.id = 'typing';

  const dots = `<div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>`;
  const local = anyLocalSelected();
  const slow = local && HARDWARE && HARDWARE.tier === 'cpu';

  let progress = '';
  if (local) {
    // A local run can take minutes with no output at all; without a ticking
    // clock and a progress readout it is indistinguishable from a hung server.
    const warn = slow
      ? 'Running locally on CPU — this can take several minutes.'
      : 'Running locally.';
    progress = `<div class="local-progress">
        <span id="localMsg">${warn}</span>
        <span class="elapsed" id="localElapsed">0:00</span>
        <button class="local-cancel" id="localCancel" type="button">Stop waiting</button>
        <div id="localDetail"></div>
      </div>`;
  }

  div.innerHTML = `<div class="avatar">🏥</div><div class="bubble">${dots}${progress}</div>`;
  messages.appendChild(div);
  scrollBottom();

  if (local) startLocalProgress();
  return div;
}

function startLocalProgress() {
  const started = Date.now();
  const elapsedEl = document.getElementById('localElapsed');
  const detailEl = document.getElementById('localDetail');

  localTimer = setInterval(() => {
    const s = Math.floor((Date.now() - started) / 1000);
    if (elapsedEl) elapsedEl.textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
  }, 1000);

  const poll = async () => {
    try {
      const r = await fetch('/api/local-status');
      const d = await r.json();
      const h = d.airllm;
      if (!h || !detailEl) return;
      if (h.state === 'loading' || h.state === 'splitting') {
        const total = h.layers_total || 0;
        const done = h.layers_split || 0;
        detailEl.textContent = total
          ? `Loading ${h.model || ''} — layer ${done}/${total}`
          : `Loading ${h.model || ''}…`;
      } else if (h.state === 'generating') {
        detailEl.textContent = `Generating — ${h.tokens_generated || 0}/${h.tokens_requested || 0} tokens`;
      } else {
        detailEl.textContent = '';
      }
    } catch (_) { /* sidecar may not be the active provider */ }
  };
  poll();
  localPoll = setInterval(poll, 5000);

  document.getElementById('localCancel')?.addEventListener('click', () => {
    // Stops us waiting; the sidecar keeps generating (it has no cancel API).
    activeController?.abort();
  });
}

function stopLocalProgress() {
  if (localTimer) { clearInterval(localTimer); localTimer = null; }
  if (localPoll) { clearInterval(localPoll); localPoll = null; }
}

function removeTyping() {
  stopLocalProgress();
  document.getElementById('typing')?.remove();
}

function buildGuardrailBanner(guardrail) {
  if (!guardrail || !guardrail.redacted || !(guardrail.findings || []).length) return '';
  const items = guardrail.findings
    .map(f => `${escHtml(f.entity_type)} ×${f.count}`)
    .join(', ');
  return `<div class="phi-banner">🔒 PHI de-identified before analysis: ${items}</div>`;
}

function buildProvidersCaption(providersUsed, modelsUsed) {
  if (!providersUsed) return '';
  const part = step => {
    const prov = escHtml(providerLabel(providersUsed[step]));
    const model = modelsUsed?.[step];
    return model ? `${prov} <span class="model-name">· ${escHtml(model)}</span>` : prov;
  };
  return `<div class="providers-caption">extract: ${part('extract')}`
    + ` &nbsp;|&nbsp; select: ${part('select')}`
    + ` &nbsp;|&nbsp; explain: ${part('explain')}</div>`;
}

function buildParseWarning(parseFailures) {
  if (!parseFailures || !parseFailures.length) return '';
  const steps = parseFailures.map(f => escHtml(f.step)).join(', ');
  const fellBack = parseFailures.some(f => f.fallback);
  return `<div class="parse-warning">⚠ The selected model did not return valid JSON for: ${steps}.`
    + (fellBack ? ' Showing top search matches instead of model-selected codes.' : '')
    + ` Try a larger model.</div>`;
}

function addBotMessage(text, icd10Codes, hccResult, conditionsFound, guardrail,
                       providersUsed, modelsUsed, parseFailures) {
  const div = document.createElement('div');
  div.className = 'message bot';

  let inner = `<div class="avatar">🏥</div><div class="bubble">`;
  inner += buildGuardrailBanner(guardrail);
  inner += buildParseWarning(parseFailures);
  inner += `<div>${markdownToHtml(text)}</div>`;

  // Result cards
  if ((icd10Codes && icd10Codes.length) || (hccResult && hccResult.active_hccs?.length)) {
    inner += `<div class="result-cards">`;

    // ICD-10 table
    if (icd10Codes && icd10Codes.length) {
      inner += buildIcd10Card(icd10Codes);
    }

    // HCC flags
    if (hccResult) {
      if (hccResult.active_hccs && hccResult.active_hccs.length) {
        inner += buildHccCard(hccResult);
      }
      inner += buildScoreCard(hccResult);
    }

    inner += `</div>`;
  }

  inner += buildProvidersCaption(providersUsed, modelsUsed);
  inner += `</div>`;
  div.innerHTML = inner;
  messages.appendChild(div);
  scrollBottom();
}

/* ── Card builders ──────────────────────────────────────────────────────────── */
function buildIcd10Card(codes) {
  let rows = codes.map(c => {
    const conf = c.confidence != null ? c.confidence : c.similarity ?? 0;
    const pct = Math.round(conf * 100);
    return `<tr>
      <td><span class="code-pill">${escHtml(c.icd10 || '')}</span></td>
      <td>${escHtml(c.description || '')}</td>
      <td><span class="cc-pill">CC${escHtml(c.cc || '')}</span></td>
      <td>${escHtml(c.condition || '')}</td>
      <td>
        <div class="conf-bar">
          <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:${pct}%"></div></div>
          <span style="font-size:.75rem;color:#64748b">${pct}%</span>
        </div>
      </td>
    </tr>`;
  }).join('');

  return `<div class="card">
    <div class="card-header">🏷 ICD-10 Codes Identified (${codes.length})</div>
    <table class="icd-table">
      <thead><tr>
        <th>Code</th><th>Description</th><th>CC</th><th>Condition</th><th>Confidence</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}

function buildHccCard(hccResult) {
  const hccs = hccResult.active_hccs || [];
  const interactions = hccResult.interactions_triggered || [];
  const cats = hccResult.diag_categories_triggered || [];

  let pills = hccs.map(h =>
    `<span class="hcc-pill" title="${escHtml(h.description || '')}">${escHtml(h.hcc)}</span>`
  ).join('');

  let intPills = interactions.map(i =>
    `<span class="interaction-pill" title="Comorbidity interaction">⚡ ${escHtml(i)}</span>`
  ).join('');

  let catPills = cats.map(c =>
    `<span class="hcc-pill" style="background:#fef3c7;color:#92400e;border-color:#fde68a">${escHtml(c)}</span>`
  ).join('');

  return `<div class="card">
    <div class="card-header">🧩 Active HCCs &amp; Interactions</div>
    <div class="hcc-grid">
      ${pills}
      ${catPills}
      ${intPills || ''}
      ${!pills && !intPills ? '<span style="color:#94a3b8;font-size:.82rem">No HCCs mapped</span>' : ''}
    </div>
  </div>`;
}

function buildScoreCard(hccResult) {
  const ceScores = hccResult.ce_scores || {};
  const neScores = hccResult.ne_scores || {};
  const applicable = hccResult.applicable_model || '';
  const dem = hccResult.demographics_derived || {};

  const modelLabels = {
    COMMUNITY_NA:  'Community – Non-Dual Aged',
    COMMUNITY_PBA: 'Community – Partial-Dual Aged',
    COMMUNITY_FBA: 'Community – Full-Dual Aged',
    COMMUNITY_ND:  'Community – Non-Dual Disabled',
    COMMUNITY_PBD: 'Community – Partial-Dual Disabled',
    COMMUNITY_FBD: 'Community – Full-Dual Disabled',
    INSTITUTIONAL: 'Long-Term Institutional',
    NEW_ENROLLEE:  'New Enrollee',
    SNP_NEW_ENROLLEE: 'SNP New Enrollee',
  };

  function scoreClass(v) {
    if (v < 0.7)  return 'low';
    if (v < 1.8)  return 'mid';
    return 'high';
  }

  const allScores = { ...ceScores, ...neScores };
  let rows = Object.entries(allScores).map(([k, v]) => {
    const isActive = k === applicable;
    const tag = isActive ? '<span class="active-tag">▶ Applies</span>' : '';
    const cls = scoreClass(v);
    return `<tr class="${isActive ? 'active-row' : ''}">
      <td>${modelLabels[k] || k}${tag}</td>
      <td><span class="score-val ${cls}">${v.toFixed(3)}</span></td>
      <td><span style="font-size:.75rem;color:#94a3b8">${v >= 1 ? '+' : ''}${((v - 1) * 100).toFixed(1)}% vs avg</span></td>
    </tr>`;
  }).join('');

  const demStr = dem.age
    ? `Age ${dem.age} · ${dem.sex_label} · ${dem.orec_label}`
    : '';

  return `<div class="card">
    <div class="card-header">📊 HCC Risk Scores – Payment Year 2027
      ${demStr ? `<span style="font-weight:400;color:#64748b;text-transform:none;font-size:.75rem">(${escHtml(demStr)})</span>` : ''}
    </div>
    <table class="score-table">
      <thead><tr><th>Model</th><th>Risk Score</th><th>vs Average</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    <div style="padding:8px 12px;font-size:.73rem;color:#64748b;border-top:1px solid #f1f5f9">
      Average Medicare beneficiary = 1.000 · Score &lt;0.7 = low risk · &gt;1.8 = high risk
    </div>
  </div>`;
}

/* ── Send ───────────────────────────────────────────────────────────────────── */
async function send() {
  const text = msgInput.value.trim();
  if (!text) return;

  msgInput.value = '';
  msgInput.style.height = '';
  sendBtn.disabled = true;

  addUserMessage(text);
  const typingEl = addTyping();

  // No timeout: a local model can legitimately take minutes. The controller
  // exists so the user can stop waiting via the Cancel button.
  activeController = new AbortController();

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: activeController.signal,
      body: JSON.stringify({
        message: text,
        demographics: getDemographics(),
        conversation_history: conversationHistory,
        providers: getProviders(),
        models: getModels(),
      }),
    });

    // Without this a 502 fell through to `data.response || '(no response)'`,
    // hiding the provider error entirely.
    if (!resp.ok) {
      let detail = `HTTP ${resp.status}`;
      try {
        const e = await resp.json();
        detail = e.detail || e.error?.message || detail;
      } catch (_) { /* non-JSON error body */ }
      removeTyping();
      addBotMessage(`⚠️ ${detail}`, [], null, [], null, null, null, null);
      sendBtn.disabled = !systemReady;
      activeController = null;
      return;
    }

    const data = await resp.json();
    removeTyping();

    conversationHistory = data.conversation_history || conversationHistory;
    addBotMessage(
      data.response || '(no response)',
      data.icd10_codes || [],
      data.hcc_result || null,
      data.conditions_found || [],
      data.guardrail || null,
      data.providers_used || null,
      data.models_used || null,
      data.parse_failures || null,
    );
  } catch (err) {
    removeTyping();
    if (err.name === 'AbortError') {
      addBotMessage('Stopped waiting. The local model may still be generating in the background —'
        + ' check the sidecar console.', [], null, []);
    } else {
      addBotMessage(`⚠️ Error: ${err.message}. Make sure the backend is running.`, [], null, []);
    }
  }

  activeController = null;
  sendBtn.disabled = !systemReady;
}

sendBtn.addEventListener('click', send);
msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
});

/* Auto-resize textarea */
msgInput.addEventListener('input', () => {
  msgInput.style.height = '';
  msgInput.style.height = Math.min(msgInput.scrollHeight, 160) + 'px';
});

/* Clear */
clearBtn.addEventListener('click', () => {
  conversationHistory = [];
  messages.innerHTML = '';
});

/* ── Utilities ──────────────────────────────────────────────────────────────── */
function escHtml(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function markdownToHtml(md) {
  // Very lightweight markdown: bold, code, line breaks, paragraphs
  let html = escHtml(md)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/`([^`]+)`/g, '<code style="background:#f1f5f9;padding:1px 5px;border-radius:4px;font-size:.85em">$1</code>')
    .replace(/\n\n/g, '</p><p>')
    .replace(/\n/g, '<br>');
  // Handle numbered lists
  html = html.replace(/((?:^\d+\..+<br>)+)/gm, '<ol style="padding-left:18px;margin:4px 0">$1</ol>');
  html = html.replace(/^(\d+)\. (.+)/gm, '<li>$2</li>');
  // Handle bullet lists
  html = html.replace(/^[-*] (.+)/gm, '<li>$1</li>');
  return `<p>${html}</p>`;
}
