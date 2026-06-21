import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  fetchJudgeStats,
  fetchJudgeViolations,
  connectJudgeWebSocket,
  buildJudgeImageUrl,
} from '../api/judgeClient';
import './JudgeFeed.css';

export default function JudgeFeed() {
  const [stats, setStats] = useState({});
  const [violations, setViolations] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const wsRef = useRef(null);
  const pollRef = useRef(null);

  const fetchData = useCallback(async () => {
    try {
      const [statsData, violationsData] = await Promise.all([
        fetchJudgeStats(),
        fetchJudgeViolations(50, 0),
      ]);
      setStats(statsData);
      setViolations(violationsData.violations || []);
      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    pollRef.current = setInterval(fetchData, 3000);
    return () => clearInterval(pollRef.current);
  }, [fetchData]);

  // WebSocket connection
  useEffect(() => {
    const conn = connectJudgeWebSocket(
      (statsUpdate, violationEvent) => {
        if (statsUpdate) {
          setStats(prev => ({ ...prev, ...statsUpdate }));
        }
        if (violationEvent) {
          // A new violation arrived live — refresh data
          fetchData();
        }
        setWsConnected(true);
      },
      () => {
        setWsConnected(false);
      }
    );
    wsRef.current = conn;

    // Check connection state periodically
    const stateCheck = setInterval(() => {
      if (conn.getState && conn.getState() === WebSocket.OPEN) {
        setWsConnected(true);
      } else {
        setWsConnected(false);
      }
    }, 2000);

    return () => {
      clearInterval(stateCheck);
      conn.close();
    };
  }, [fetchData]);

  const statEntries = Object.entries(stats).filter(([key]) => key !== 'total');
  const total = stats.total || 0;

  const formatType = (type) => type.replace(/_/g, ' ');

  const formatTimestamp = (ts) => {
    if (!ts) return '—';
    try {
      const d = new Date(ts);
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return ts;
    }
  };

  return (
    <div className="page-container">
      <h2 className="page-title">LLM Judge Feed</h2>
      <p className="judge-description">
        This feed shows violations confirmed by the local vision-language model (Ollama) after
        initial detection — run <code>edge_client.py</code> against a video file to populate it.
      </p>

      {/* Connection status */}
      <div className="connection-bar">
        <div className={`conn-indicator ${wsConnected ? 'connected' : 'disconnected'}`}>
          <div className="conn-dot"></div>
          <span>{wsConnected ? 'Live — connected to judge server' : 'Disconnected — judge server not reachable'}</span>
        </div>
        {!wsConnected && !loading && (
          <span className="conn-hint">
            Start the judge server with: <code>uvicorn server:app --port 8001 --app-dir hybrid_mvp</code>
          </span>
        )}
      </div>

      {/* KPI row */}
      <div className="judge-kpi-row">
        <div className="judge-kpi-card judge-kpi-total">
          <div className="judge-kpi-value">{total}</div>
          <div className="judge-kpi-label">Total Confirmed</div>
        </div>
        {statEntries.map(([type, count]) => (
          <div className="judge-kpi-card" key={type}>
            <div className="judge-kpi-value">{count}</div>
            <div className="judge-kpi-label">{formatType(type)}</div>
          </div>
        ))}
        {statEntries.length === 0 && !loading && (
          <div className="judge-kpi-card judge-kpi-empty">
            <div className="judge-kpi-value">—</div>
            <div className="judge-kpi-label">No violations yet</div>
          </div>
        )}
      </div>

      {/* Violation feed */}
      <div className="judge-feed-section">
        <div className="feed-header">
          <span className="feed-title">Confirmed Violations</span>
          <span className="feed-count">{violations.length} records</span>
        </div>

        {loading && (
          <div className="feed-empty-state">
            <div className="feed-empty-icon">⏳</div>
            <div>Loading judge feed…</div>
          </div>
        )}

        {!loading && error && violations.length === 0 && (
          <div className="feed-empty-state">
            <div className="feed-empty-icon">🔌</div>
            <div className="feed-empty-title">Judge server not available</div>
            <div className="feed-empty-hint">
              Start <code>hybrid_mvp/server.py</code> on port 8001, then run
              <code>edge_client.py</code> to begin processing a video.
            </div>
          </div>
        )}

        {!loading && !error && violations.length === 0 && (
          <div className="feed-empty-state">
            <div className="feed-empty-icon">📭</div>
            <div className="feed-empty-title">No confirmed violations yet</div>
            <div className="feed-empty-hint">
              Run <code>python hybrid_mvp/edge_client.py</code> against a video file to
              populate this feed with LLM-confirmed violations.
            </div>
          </div>
        )}

        {violations.length > 0 && (
          <div className="violation-grid">
            {violations.map((v) => (
              <div className="violation-card" key={v.id}>
                <div className="vc-image-wrapper">
                  {v.image_url ? (
                    <img
                      src={buildJudgeImageUrl(v.image_url)}
                      alt={`${v.violation_type} evidence`}
                      className="vc-image"
                      loading="lazy"
                      onError={(e) => { e.target.style.display = 'none'; }}
                    />
                  ) : (
                    <div className="vc-image-placeholder">No image</div>
                  )}
                </div>
                <div className="vc-details">
                  <div className="vc-type-badge">
                    {formatType(v.violation_type)}
                  </div>
                  <div className="vc-meta">
                    <span className="vc-meta-item">
                      <span className="vc-meta-label">Vehicle</span>
                      <span className="vc-meta-value">{v.vehicle_id}</span>
                    </span>
                    <span className="vc-meta-item">
                      <span className="vc-meta-label">Frame</span>
                      <span className="vc-meta-value">{v.frame_id ?? '—'}</span>
                    </span>
                    <span className="vc-meta-item">
                      <span className="vc-meta-label">Time</span>
                      <span className="vc-meta-value">{formatTimestamp(v.timestamp)}</span>
                    </span>
                  </div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
