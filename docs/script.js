'use strict';

// All chart values come from the manuscript's preference_*.dat files.
// Bundled JS data keeps both file:// previews and GitHub Pages working offline.
const preferenceData = window.OPD_PREFERENCE_DATA;
const explorer = document.querySelector('#preference-explorer');
const chartContainer = document.querySelector('#preference-charts');
const datasetSelect = document.querySelector('#dataset');
let currentSetting = 'base8k';
const metrics = [
  {key: 'correct_rate', label: 'Correct answers', unit: '%', max: 55, ticks: [0, 25, 50]},
  {key: 'mean_response_tokens', label: 'Response length', unit: ' tokens', max: 16000, ticks: [0, 8000, 16000]},
  {key: 'repetition_rate', label: 'Severe repetition', unit: '%', max: 50, ticks: [0, 25, 50]},
];
const formatValue = (value, metric) => metric.unit === '%' ? `${value.toFixed(1)}%` : `${Math.round(value).toLocaleString('en-US')} tokens`;

function renderPreferenceCharts() {
  const dataset = datasetSelect.value;
  const color = currentSetting === 'justrl' ? '#287867' : '#b25543';
  const context = document.querySelector('#chart-context');
  context.classList.toggle('success', currentSetting === 'justrl');
  context.textContent = currentSetting === 'justrl'
    ? 'JustRL → DS-Distill-1.5B: higher preference, more correct answers, no severe repetition.'
    : 'Qwen3-4B → Qwen3-1.7B-Base: the most-preferred group is highly repetitive, with correctness near zero.';
  chartContainer.innerHTML = metrics.map((metric, index) => {
    const rows = preferenceData[currentSetting][metric.key];
    const W = 290, H = 205, left = 35, right = 16, top = 26, bottom = 28;
    const x = i => left + i * (W - left - right) / 9;
    const y = v => H - bottom - Math.min(metric.max, Math.max(0, v)) / metric.max * (H - top - bottom);
    const grid = metric.ticks.map(t => `<line x1="${left}" x2="${W-right}" y1="${y(t)}" y2="${y(t)}" stroke="#e8e9e2"/><text x="${left-8}" y="${y(t)+3}" text-anchor="end">${t >= 1000 ? `${t/1000}k` : t}</text>`).join('');
    const errors = rows.map((row, i) => {
      const low = y(row[dataset] - row[`${dataset}_errminus`]);
      const high = y(row[dataset] + row[`${dataset}_errplus`]);
      return `<path d="M${x(i)},${low}V${high}M${x(i)-2},${low}H${x(i)+2}M${x(i)-2},${high}H${x(i)+2}" stroke="${color}" opacity=".35" fill="none"/>`;
    }).join('');
    const points = rows.map((row, i) => `${x(i)},${y(row[dataset])}`).join(' ');
    const dots = rows.map((row, i) => `<circle cx="${x(i)}" cy="${y(row[dataset])}" r="${i === 9 ? 4.5 : 3}" fill="${color}"><title>A${i+1}: ${formatValue(row[dataset], metric)}</title></circle>`).join('');
    const last = rows[9][dataset];
    const tickLabels = [0, 3, 6, 9].map(i => `<text x="${x(i)}" y="${H-9}" text-anchor="middle">A${i+1}</text>`).join('');
    const teacher = metric.key === 'mean_response_tokens'
      ? `<line x1="${left}" x2="${W-right}" y1="${y(rows[0][`${dataset}_teacher`])}" y2="${y(rows[0][`${dataset}_teacher`])}" stroke="#898d80" stroke-dasharray="4 4"><title>Teacher mean: ${formatValue(rows[0][`${dataset}_teacher`], metric)}</title></line>` : '';
    const titleId = `metric-title-${index}`;
    return `<div class="mini-chart"><h4>${metric.label}${metric.unit === '%' ? ' (%)' : ' (tokens)'}</h4><svg viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="${titleId}"><title id="${titleId}">${metric.label} by teacher preference on ${dataset.toUpperCase()}. Least-preferred group: ${formatValue(rows[0][dataset], metric)}. Most-preferred group: ${formatValue(last, metric)}.</title>${grid}${teacher}${errors}<polyline points="${points}" fill="none" stroke="${color}" stroke-width="2.3" stroke-linejoin="round"/>${dots}${tickLabels}<text class="chart-value" x="${x(9)}" y="${Math.max(15, y(last)-13)}" text-anchor="end" style="fill:${color}">${formatValue(last, metric)}</text></svg></div>`;
  }).join('');
  const rows = preferenceData[currentSetting].correct_rate;
  document.querySelector('#chart-values').innerHTML = `<table><caption>${currentSetting === 'justrl' ? 'JustRL' : 'Qwen3-4B'} · ${dataset.toUpperCase()} · Group means</caption><thead><tr><th scope="col">Group</th>${metrics.map(m=>`<th scope="col">${m.label}</th>`).join('')}</tr></thead><tbody>${rows.map((_, i)=>`<tr><th scope="row">A${i+1}</th>${metrics.map(m=>`<td>${formatValue(preferenceData[currentSetting][m.key][i][dataset], m)}</td>`).join('')}</tr>`).join('')}</tbody></table>`;
}

if (preferenceData) {
  explorer.hidden = false;
  renderPreferenceCharts();
  datasetSelect.addEventListener('change', renderPreferenceCharts);
  document.querySelectorAll('[data-setting]').forEach(button => {
    button.addEventListener('click', () => {
      currentSetting = button.dataset.setting;
      document.querySelectorAll('[data-setting]').forEach(other => other.setAttribute('aria-pressed', String(other === button)));
      renderPreferenceCharts();
    });
  });
}

// Links remain ordinary image links when dialog support or JavaScript is absent.
const dialog = document.querySelector('#image-dialog');
if (typeof dialog.showModal === 'function') {
  document.querySelectorAll('a.zoomable').forEach(link => {
    link.addEventListener('click', event => {
      if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
      event.preventDefault();
      const img = link.querySelector('img');
      const expandedImage = document.querySelector('#dialog-image');
      expandedImage.src = link.href;
      expandedImage.alt = img.alt;
      document.querySelector('#dialog-caption').textContent = img.alt;
      dialog.showModal();
    });
  });
  document.querySelector('#close-dialog').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  });
}

const copyButton = document.querySelector('#copy-citation');
copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  const text = document.querySelector('#bibtex').textContent;
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(text);
    status.textContent = 'BibTeX copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C (or ⌘C) to copy.';
  }
});

// Offer a still version for motion sensitivity and let readers restart each scene.
document.querySelectorAll('.overview-animation').forEach(figure => {
  const img = figure.querySelector('img');
  const source = figure.querySelector('source');
  const button = figure.querySelector('.animation-toggle');
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  let playing = !preference.matches;
  function update() {
    const url = playing ? img.dataset.animation : img.dataset.poster;
    source.srcset = url;
    img.src = url;
    button.textContent = playing ? 'Show still image' : 'Play animation';
    button.setAttribute('aria-label', `${button.textContent}: ${figure.querySelector('h3').textContent}`);
  }
  button.hidden = false;
  update();
  button.addEventListener('click', () => { playing = !playing; update(); });
  preference.addEventListener('change', event => { playing = !event.matches; update(); });
});
