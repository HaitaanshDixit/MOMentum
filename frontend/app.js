const API = window.location.origin;

let selectedFile = null;
let selectedFormat = 'md';

function toggleTheme() {
  const body = document.body;
  const icon = document.getElementById('theme-icon');
  if (body.classList.contains('dark')) {
    body.classList.replace('dark', 'light');
    icon.textContent = '🌙';
    localStorage.setItem('theme', 'light');
  } else {
    body.classList.replace('light', 'dark');
    icon.textContent = '☀️';
    localStorage.setItem('theme', 'dark');
  }
}

function initTheme() {
  const saved = localStorage.getItem('theme') || 'dark';
  document.body.className = saved;
  document.getElementById('theme-icon').textContent = saved === 'dark' ? '☀️' : '🌙';
}

function setupDragDrop() {
  const zone = document.getElementById('drop-zone');
  const input = document.getElementById('file-input');

  zone.addEventListener('click', () => input.click());
  input.addEventListener('change', () => {
    if (input.files[0]) setFile(input.files[0]);
  });

  zone.addEventListener('dragover', e => {
    e.preventDefault();
    zone.classList.add('dragover');
  });
  zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));
  zone.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('dragover');
    if (e.dataTransfer.files[0]) setFile(e.dataTransfer.files[0]);
  });
}

function setFile(file) {
  selectedFile = file;
  document.getElementById('drop-zone').classList.add('hidden');
  document.getElementById('file-selected').classList.remove('hidden');
  document.getElementById('file-name-display').textContent = file.name;
  document.getElementById('file-size-display').textContent = formatBytes(file.size);
  document.getElementById('generate-btn').disabled = false;
}

function clearFile() {
  selectedFile = null;
  document.getElementById('drop-zone').classList.remove('hidden');
  document.getElementById('file-selected').classList.add('hidden');
  document.getElementById('file-input').value = '';
  document.getElementById('generate-btn').disabled = true;
}

function selectFormat(btn) {
  document.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  selectedFormat = btn.dataset.format;
}

function formatBytes(bytes) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

const STAGES = [
  [10, 'Pre-processing audio...'],
  [25, 'Transcribing with Whisper...'],
  [55, 'Generating summary...'],
  [75, 'Review agent checking quality...'],
  [90, 'Formatting and exporting...'],
  [97, 'Indexing for search...'],
];

let progressInterval = null;
let currentStageIdx = 0;

function startProgress() {
  currentStageIdx = 0;
  setProgress(0, 'Starting pipeline...');
  progressInterval = setInterval(() => {
    if (currentStageIdx < STAGES.length) {
      const [pct, label] = STAGES[currentStageIdx];
      setProgress(pct, label);
      currentStageIdx++;
    }
  }, 4000);
}

function setProgress(pct, label) {
  document.getElementById('progress-bar').style.width = `${pct}%`;
  document.getElementById('progress-stage').textContent = label;
}

function stopProgress() {
  clearInterval(progressInterval);
  setProgress(100, 'Done!');
}

async function generateMOM() {
  if (!selectedFile) return;

  const title = document.getElementById('meeting-title').value.trim();
  const saveTranscript = document.getElementById('save-transcript').checked;
  const allFormats = document.getElementById('all-formats').checked;
  const format = allFormats ? 'md' : selectedFormat;

  hide('upload-area');
  hide('result-card');
  hide('error-card');
  show('progress-card');
  document.getElementById('progress-title').textContent = `Processing ${selectedFile.name}...`;
  startProgress();

  const formData = new FormData();
  formData.append('file', selectedFile);
  formData.append('format', format);
  formData.append('save_transcript_flag', saveTranscript);
  if (title) formData.append('title', title);

  try {
    const res = await fetch(`${API}/api/upload`, {
      method: 'POST',
      body: formData,
    });

    stopProgress();

    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || 'Something went wrong.');
    }

    const data = await res.json();
    showResult(data);

  } catch (err) {
    stopProgress();
    showError(err.message);
  }
}

function showResult(data) {
  hide('progress-card');
  show('result-card');

  document.getElementById('result-meta').textContent =
    `${data.duration} · ${data.word_count} words · ${data.language} · Review score ${data.review_score}/100`;

  document.getElementById('result-overview').textContent =
    data.preview.overview || 'No overview generated.';

  renderList('result-actions', data.preview.action_items, 'checklist');
  renderList('result-decisions', data.preview.decisions, 'plain-list');
  renderList('result-nextsteps', data.preview.next_steps, 'plain-list');

  const dlBtn = document.getElementById('download-btn');
  dlBtn.href = `${API}${data.download_url}`;
  dlBtn.textContent = `Download ${data.filename.split('.').pop().toUpperCase()}`;
}

function renderList(id, items, cls) {
  const el = document.getElementById(id);
  if (!items || items.length === 0) {
    el.innerHTML = '<li style="color:var(--text-3);list-style:none;padding:4px 0">None identified</li>';
    return;
  }
  el.className = cls;
  el.innerHTML = items.map(i => `<li>${i}</li>`).join('');
}

function showError(msg) {
  hide('progress-card');
  show('error-card');
  document.getElementById('error-msg').textContent = msg;
}

function resetUpload() {
  clearFile();
  hide('progress-card');
  hide('result-card');
  hide('error-card');
  show('upload-area');
  document.getElementById('meeting-title').value = '';
}

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

document.addEventListener('DOMContentLoaded', () => {
  initTheme();
  setupDragDrop();
});