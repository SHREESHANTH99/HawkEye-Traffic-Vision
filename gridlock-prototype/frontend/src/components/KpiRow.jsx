import React from 'react';
import './KpiRow.css';

export default function KpiRow({ summary, framesProcessed }) {
  const total = summary?.total || 0;
  const helmet = summary?.NO_HELMET || 0;
  const triple = summary?.TRIPLE_RIDING || 0;

  const helmetPct = total > 0 ? (helmet / total) * 100 : 0;
  const triplePct = total > 0 ? (triple / total) * 100 : 0;

  return (
    <div className="kpi-container">
      <div className="kpi-telemetry">
        <span className="telemetry-label">SYSTEM_TELEMETRY // FRAMES_PROCESSED</span>
        <span className="telemetry-value mono">{framesProcessed || 0}</span>
      </div>
      
      <div className="kpi-main-block">
        <div className="kpi-primary">
          <div className="kpi-primary-label">TOTAL VIOLATIONS</div>
          <div className="kpi-primary-value mono">{total}</div>
        </div>
        
        <div className="kpi-subordinates">
          <div className="kpi-sub-item">
            <div className="kpi-sub-header">
              <span className="kpi-sub-label">NO HELMET</span>
              <span className="kpi-sub-value mono text-error">{helmet}</span>
            </div>
            <div className="kpi-sub-bar-bg">
              <div className="kpi-sub-bar-fill bg-error" style={{ width: `${helmetPct}%` }}></div>
            </div>
          </div>
          
          <div className="kpi-sub-item">
            <div className="kpi-sub-header">
              <span className="kpi-sub-label">TRIPLE RIDING</span>
              <span className="kpi-sub-value mono text-warning">{triple}</span>
            </div>
            <div className="kpi-sub-bar-bg">
              <div className="kpi-sub-bar-fill bg-warning" style={{ width: `${triplePct}%` }}></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
