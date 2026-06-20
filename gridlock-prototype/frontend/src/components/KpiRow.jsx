import React from 'react';
import './KpiRow.css';

export default function KpiRow({ summary, framesProcessed }) {
  return (
    <div className="kpi-row">
      <div className="kpi-card">
        <div className="kpi-content">
          <div className="kpi-value">{framesProcessed || 0}</div>
          <div className="kpi-label">Frames Processed</div>
        </div>
        <div className="kpi-deco icon-search"></div>
      </div>
      
      <div className="kpi-card">
        <div className="kpi-content">
          <div className="kpi-value highlight">{summary?.total || 0}</div>
          <div className="kpi-label">Total Violations</div>
        </div>
        <div className="kpi-deco icon-alert"></div>
      </div>
      
      <div className="kpi-card">
        <div className="kpi-content">
          <div className="kpi-value text-error">{summary?.NO_HELMET || 0}</div>
          <div className="kpi-label">No Helmet</div>
        </div>
        <div className="kpi-deco icon-helmet"></div>
      </div>
      
      <div className="kpi-card">
        <div className="kpi-content">
          <div className="kpi-value text-warning">{summary?.TRIPLE_RIDING || 0}</div>
          <div className="kpi-label">Triple Riding</div>
        </div>
        <div className="kpi-deco icon-bike"></div>
      </div>
    </div>
  );
}
