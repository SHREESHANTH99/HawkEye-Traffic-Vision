import React from 'react';
import KpiRow from '../components/KpiRow';
import ViolationLog from '../components/ViolationLog';
import { useSettings } from '../contexts/SettingsContext';

export default function Dashboard() {
  const { summary, framesProcessed, violations, handleClearLog } = useSettings();

  return (
    <div className="page-container">
      <h2 className="page-title">Dashboard Overview</h2>
      <KpiRow summary={summary} framesProcessed={framesProcessed} />
      <div className="panels-container">
        <ViolationLog violations={violations} onClear={handleClearLog} />
      </div>
    </div>
  );
}
