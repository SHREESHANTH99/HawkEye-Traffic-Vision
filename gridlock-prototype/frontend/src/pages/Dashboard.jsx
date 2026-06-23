import React from 'react';
import KpiRow from '../components/KpiRow';
import ViolationLog from '../components/ViolationLog';
import { useSettings } from '../contexts/SettingsContext';
import './Dashboard.css';

export default function Dashboard() {
  const { summary, framesProcessed, violations, handleClearLog } = useSettings();

  return (
    <div className="page-container">
      <div className="dashboard-header-band">
        <h2 className="dashboard-title">DASHBOARD OVERVIEW</h2>
      </div>
      <KpiRow summary={summary} framesProcessed={framesProcessed} />
      <div className="dashboard-content">
        <ViolationLog violations={violations} onClear={handleClearLog} />
      </div>
    </div>
  );
}
