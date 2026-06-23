import React from 'react';
import DetectionPanel from '../components/DetectionPanel';
import { useSettings } from '../contexts/SettingsContext';

export default function DetectionStudio() {
  const { settings, handleViolationsUpdate } = useSettings();

  return (
    <div className="page-container">
      <h2 className="page-title govt-badge">Detection Studio</h2>
      <div className="detection-studio-layout">
        <DetectionPanel 
          settings={settings} 
          onViolationsUpdate={handleViolationsUpdate} 
        />
      </div>
    </div>
  );
}
