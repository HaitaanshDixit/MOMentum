const API = 'http://127.0.0.1:8000';

let selectedFile = null;
let selectedFormat = 'md';

function showPage(name) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`page-${name}`).classList.add('active');
  document.querySelectorAll('.nav-btn').forEach(b => {
    if (b.textContent.toLowerCase().includes(name === 'upload' ? 'gen' : name)) {
      b.classList.add('active');
    }
  });
  if (name === 'meetings') loadMeetings();
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

  // Show progress, hide others
  hide('upload-area');
  hide('result-card');
  hide('error-card');
  show('progress-card');
  document.getElementById('progress-title').textContent = `Processing ${selectedFile.name}...`;
  startProgress();

  // Build form data
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

async function doSearch() {
  const query = document.getElementById('search-input').value.trim();
  if (!query) return;

  const container = document.getElementById('search-results');
  container.innerHTML = '<p class="loading-text">Searching...</p>';

  try {
    const res = await fetch(`${API}/api/search?q=${encodeURIComponent(query)}&top_k=5`);
    const data = await res.json();

    if (!res.ok) throw new Error(data.detail || 'Search failed.');

    if (data.results.length === 0) {
      container.innerHTML = '<p class="empty-text">No relevant meetings found. Try a different query.</p>';
      return;
    }

    container.innerHTML = data.results.map(r => `
      <div class="result-item">
        <div class="result-item-header">
          <span class="result-item-title">${r.audio_file}</span>
          <span class="result-item-score">Score ${(r.score * 100).toFixed(0)}%</span>
        </div>
        <p class="result-item-meta">${r.date} · ${r.duration}</p>
        <p class="result-item-preview">${r.transcript_preview}</p>
        ${r.download_url
          ? `<a class="btn-secondary" href="${API}${r.download_url}" target="_blank">Download MOM</a>`
          : ''}
      </div>
    `).join('');

  } catch (err) {
    container.innerHTML = `<p class="empty-text">${err.message}</p>`;
  }
}

async function loadMeetings() {
  const container = document.getElementById('meetings-list');
  container.innerHTML = '<p class="loading-text">Loading meetings...</p>';

  try {
    const res = await fetch(`${API}/api/meetings`);
    const data = await res.json();

    if (data.meetings.length === 0) {
      container.innerHTML = '<p class="empty-text">No meetings indexed yet. Generate your first MOM above!</p>';
      return;
    }

    container.innerHTML = data.meetings.map(m => `
      <div class="result-item">
        <div class="result-item-header">
          <span class="result-item-title">${m.audio_file}</span>
          <span class="result-item-score">${m.duration}</span>
        </div>
        <p class="result-item-meta">${m.date} · ${m.word_count} words</p>
        <p class="result-item-preview">${m.transcript_preview}</p>
        ${m.download_url
          ? `<a class="btn-secondary" href="${API}${m.download_url}" target="_blank">Download MOM</a>`
          : ''}
      </div>
    `).join('');

  } catch (err) {
    container.innerHTML = `<p class="empty-text">Could not load meetings. Is the server running?</p>`;
  }
}

function show(id) { document.getElementById(id).classList.remove('hidden'); }
function hide(id) { document.getElementById(id).classList.add('hidden'); }

document.addEventListener('DOMContentLoaded', () => {
  setupDragDrop();
});