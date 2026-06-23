import React from 'react';
import './KpiRow.css';

export default function KpiRow({ summary, framesProcessed }) {
  const total = summary?.total || 0;
  const helmet = summary?.NO_HELMET || 0;
  const triple = summary?.TRIPLE_RIDING || 0;

  return (
    <div className="kpi-container">
      <div className="kpi-strip">
        <div className="kpi-block">
          <div className="kpi-value mono">{total}</div>
          <div className="kpi-label govt-badge">Total Violations</div>
        </div>
        
        <div className="kpi-block">
          <div className="kpi-value mono">{helmet}</div>
          <div className="kpi-label govt-badge">No Helmet</div>
        </div>
        
        <div className="kpi-block">
          <div className="kpi-value mono">{triple}</div>
          <div className="kpi-label govt-badge">Triple Riding</div>
        </div>

        <div className="kpi-block">
          <div className="kpi-value mono">{framesProcessed || 0}</div>
          <div className="kpi-label govt-badge">Frames Processed</div>
        </div>
      </div>
    </div>
  );
}
