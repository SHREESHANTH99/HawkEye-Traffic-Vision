import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  fetchJudgeStats,
  fetchJudgeViolations,
  connectJudgeWebSocket,
  buildJudgeImageUrl,
} from '../api/judgeClient';
import EvidenceRecord from '../components/EvidenceRecord';
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
          <div className="feed-empty-state mono">
            <div className="feed-empty-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
            </div>
            <div className="feed-empty-title text-error">[ CONNECTION_FAILED ]</div>
            <div className="feed-empty-hint">
              &gt; Judge server not available. <br/>
              &gt; Start <code>hybrid_mvp/server.py</code> on port 8001.
            </div>
          </div>
        )}

        {!loading && !error && violations.length === 0 && (
          <div className="feed-empty-state mono">
            <div className="feed-empty-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="4 17 10 11 4 5"></polyline><line x1="12" y1="19" x2="20" y2="19"></line></svg>
            </div>
            <div className="feed-empty-title">[ SYSTEM_LISTENING ]</div>
            <div className="feed-empty-hint">
              &gt; Awaiting evidence from edge client...<br/>
              &gt; Run <code>python hybrid_mvp/edge_client.py</code> to stream violations.
            </div>
          </div>
        )}

        {violations.length > 0 && (
          <div className="violation-grid">
            {violations.map((v) => {
              const record = {
                id: v.id,
                type: v.violation_type,
                plateText: v.plate_text,
                plateValid: v.plate_valid,
                timestamp: v.timestamp,
                frameId: v.frame_id,
                vehicleId: v.vehicle_id,
                imageUrl: v.image_url ? buildJudgeImageUrl(v.image_url) : null
              };
              return <EvidenceRecord key={v.id} record={record} variant="full" />;
            })}
          </div>
        )}
      </div>
    </div>
  );
}
