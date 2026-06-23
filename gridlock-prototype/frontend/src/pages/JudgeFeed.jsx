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
          fetchData();
        }
        setWsConnected(true);
      },
      () => {
        setWsConnected(false);
      }
    );
    wsRef.current = conn;

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

  return (
    <div className="page-container">
      <div className="dashboard-header-band">
        <h2 className="dashboard-title">LLM JUDGE FEED</h2>
      </div>

      {/* Connection status */}
      <div className={`connection-banner ${wsConnected ? 'connected' : 'disconnected'}`}>
        <div className="conn-indicator">
          <div className="conn-dot"></div>
          <span className="conn-text">{wsConnected ? 'SYSTEM ONLINE — CONNECTED TO JUDGE SERVER' : 'OFFLINE — JUDGE SERVER NOT REACHABLE'}</span>
        </div>
        {!wsConnected && !loading && (
          <span className="conn-hint">
            Ensure backend is running: <code>uvicorn hybrid_mvp.server:app --port 8001</code>
          </span>
        )}
      </div>

      {/* KPI row */}
      <div className="judge-kpi-row">
        <div className="judge-kpi-card judge-kpi-total">
          <div className="judge-kpi-value mono">{total}</div>
          <div className="judge-kpi-label govt-badge">Total Confirmed</div>
        </div>
        {statEntries.map(([type, count]) => (
          <div className="judge-kpi-card" key={type}>
            <div className="judge-kpi-value mono">{count}</div>
            <div className="judge-kpi-label govt-badge">{formatType(type)}</div>
          </div>
        ))}
        {statEntries.length === 0 && !loading && (
          <div className="judge-kpi-card judge-kpi-empty">
            <div className="judge-kpi-value mono">—</div>
            <div className="judge-kpi-label govt-badge">No violations yet</div>
          </div>
        )}
      </div>

      {/* Violation feed */}
      <div className="judge-feed-section">
        <div className="feed-header">
          <span className="feed-title govt-badge">Confirmed Violations Log</span>
          <span className="feed-count">{violations.length} RECORDS</span>
        </div>

        {loading && (
          <div className="feed-empty-state">
            <div className="feed-empty-icon">⏳</div>
            <div>LOADING JUDGE FEED...</div>
          </div>
        )}

        {!loading && error && violations.length === 0 && (
          <div className="feed-empty-state mono">
            <div className="feed-empty-title text-error">[ CONNECTION_FAILED ]</div>
            <div className="feed-empty-hint">
              &gt; Judge server not available. <br/>
              &gt; Start server on port 8001.
            </div>
          </div>
        )}

        {!loading && !error && violations.length === 0 && (
          <div className="feed-empty-state mono">
            <div className="feed-empty-title">[ SYSTEM_LISTENING ]</div>
            <div className="feed-empty-hint">
              &gt; Awaiting evidence from edge client...<br/>
              &gt; Run edge_client.py to stream violations.
            </div>
          </div>
        )}

        {violations.length > 0 && (
          <div className="table-wrapper">
            <table className="violation-table">
              <thead>
                <tr>
                  <th>EVIDENCE</th>
                  <th>TYPE</th>
                  <th>PLATE</th>
                  <th>CONFIDENCE</th>
                  <th>TIME</th>
                  <th>FRAME</th>
                </tr>
              </thead>
              <tbody>
                {violations.map((v) => {
                  const record = {
                    id: v.id,
                    type: v.violation_type,
                    plateText: v.plate_text,
                    plateValid: v.plate_valid,
                    confidence: v.confidence,
                    timestamp: v.timestamp,
                    frameId: v.frame_id,
                    vehicleId: v.vehicle_id,
                    imageUrl: v.image_url ? buildJudgeImageUrl(v.image_url) : null
                  };
                  return <EvidenceRecord key={v.id} record={record} variant="full" />;
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
