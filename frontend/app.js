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

function addTyping() {
  const div = document.createElement('div');
  div.className = 'message bot';
  div.id = 'typing';
  div.innerHTML = `<div class="avatar">🏥</div>
    <div class="bubble"><div class="typing"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div></div>`;
  messages.appendChild(div);
  scrollBottom();
  return div;
}

function removeTyping() {
  document.getElementById('typing')?.remove();
}

function addBotMessage(text, icd10Codes, hccResult, conditionsFound) {
  const div = document.createElement('div');
  div.className = 'message bot';

  let inner = `<div class="avatar">🏥</div><div class="bubble">`;
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

  try {
    const resp = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: text,
        demographics: getDemographics(),
        conversation_history: conversationHistory,
      }),
    });

    const data = await resp.json();
    removeTyping();

    conversationHistory = data.conversation_history || conversationHistory;
    addBotMessage(
      data.response || '(no response)',
      data.icd10_codes || [],
      data.hcc_result || null,
      data.conditions_found || [],
    );
  } catch (err) {
    removeTyping();
    addBotMessage(`⚠️ Error: ${err.message}. Make sure the backend is running.`, [], null, []);
  }

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
