const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';
const WS_BASE = API_BASE.replace(/^http/, 'ws');

export async function checkHealth() {
  const res = await fetch(`${API_BASE}/health`);
  if (!res.ok) throw new Error('Health check failed');
  return res.json();
}

export async function detectImage(file, settings) {
  const formData = new FormData();
  formData.append('file', file);
  
  const params = new URLSearchParams({
    conf_threshold: settings.confThreshold,
    overlap_threshold: settings.overlapThreshold,
    triple_threshold: settings.tripleThreshold,
    run_alpr: true,
  });

  const res = await fetch(`${API_BASE}/detect/image?${params.toString()}`, {
    method: 'POST',
    body: formData,
  });
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || 'Detection failed');
  }
  return res.json();
}

export async function readPlate(file) {
  const formData = new FormData();
  formData.append('file', file);
  
  const res = await fetch(`${API_BASE}/alpr/read`, {
    method: 'POST',
    body: formData,
  });
  
  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || 'ALPR failed');
  }
  return res.json();
}

export async function fetchViolationLog(limit = 50) {
  const res = await fetch(`${API_BASE}/violations/log?limit=${limit}`);
  if (!res.ok) throw new Error('Failed to fetch violation log');
  return res.json();
}

export async function fetchViolationSummary() {
  const res = await fetch(`${API_BASE}/violations/summary`);
  if (!res.ok) throw new Error('Failed to fetch violation summary');
  return res.json();
}

export async function clearViolationLog() {
  const res = await fetch(`${API_BASE}/violations/log`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to clear log');
  return res.json();
}

export async function fetchRoboflowDataset(body) {
  const res = await fetch(`${API_BASE}/dataset/fetch/roboflow`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || 'Dataset fetch failed');
  }
  return res.json();
}

export async function fetchKaggleDataset(slug) {
  const res = await fetch(`${API_BASE}/dataset/fetch/kaggle?dataset_slug=${encodeURIComponent(slug)}`, {
    method: 'POST',
  });
  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new Error(error.detail || 'Dataset fetch failed');
  }
  return res.json();
}

export function streamVideoDetection(file, onFrame, onDone, onError) {
  const ws = new WebSocket(`${WS_BASE}/ws/detect/video`);
  
  ws.onopen = () => {
    ws.send(file);
  };
  
  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.error) {
        onError(new Error(data.error));
        ws.close();
      } else if (data.done) {
        onDone();
        ws.close();
      } else {
        onFrame(data);
      }
    } catch (err) {
      onError(err);
    }
  };
  
  ws.onerror = (err) => onError(err);
  ws.onclose = () => onDone();
  
  return {
    close: () => ws.close()
  };
}
