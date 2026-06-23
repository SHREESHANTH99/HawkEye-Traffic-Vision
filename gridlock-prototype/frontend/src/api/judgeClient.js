const JUDGE_API_BASE = import.meta.env.VITE_JUDGE_API_BASE || 'http://localhost:8001';
const JUDGE_WS_BASE = JUDGE_API_BASE.replace(/^http/, 'ws');

export async function fetchJudgeStats() {
  const res = await fetch(`${JUDGE_API_BASE}/api/stats`);
  if (!res.ok) throw new Error('Failed to fetch judge stats');
  return res.json();
}

export async function fetchJudgeViolations(limit = 50, offset = 0) {
  const res = await fetch(`${JUDGE_API_BASE}/api/violations?limit=${limit}&offset=${offset}`);
  if (!res.ok) throw new Error('Failed to fetch judge violations');
  return res.json();
}

export async function clearJudgeLogs() {
  const res = await fetch(`${JUDGE_API_BASE}/api/violations`, { method: 'DELETE' });
  if (!res.ok) throw new Error('Failed to clear judge logs');
  return res.json();
}

export function connectJudgeWebSocket(onStats, onError) {
  let ws;
  let pingInterval;

  try {
    ws = new WebSocket(`${JUDGE_WS_BASE}/ws`);
  } catch (err) {
    onError(err);
    return { close: () => {} };
  }

  ws.onopen = () => {
    pingInterval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send('ping');
      }
    }, 25000);
  };

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data);
      if (data.event === 'init_stats') {
        onStats(data.stats);
      } else if (data.event === 'violation') {
        // A new violation just arrived — trigger a stats refresh
        onStats(null, data);
      }
    } catch (err) {
      // pong or non-JSON messages — ignore
    }
  };

  ws.onerror = () => {
    onError(new Error('Judge WebSocket connection failed'));
  };

  ws.onclose = () => {
    clearInterval(pingInterval);
  };

  return {
    close: () => {
      clearInterval(pingInterval);
      if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
        ws.close();
      }
    },
    getState: () => ws.readyState,
  };
}

export function buildJudgeImageUrl(imagePath) {
  if (!imagePath) return '';
  // image_path from the DB is like "/static/violations/xyz.jpg"
  // If it already starts with /static, prepend the base URL
  if (imagePath.startsWith('/static')) {
    return `${JUDGE_API_BASE}${imagePath}`;
  }
  // If it's a bare filename like "violations/xyz.jpg"
  return `${JUDGE_API_BASE}/static/${imagePath}`;
}
