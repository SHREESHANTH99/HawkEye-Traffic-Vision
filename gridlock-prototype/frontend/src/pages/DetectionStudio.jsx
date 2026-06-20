import React from 'react';
import DetectionPanel from '../components/DetectionPanel';
import { useSettings } from '../contexts/SettingsContext';

export default function DetectionStudio() {
  const { settings, handleViolationsUpdate } = useSettings();

  return (
    <div className="page-container">
      <h2 className="page-title">Detection Studio</h2>
      <div className="panels-container" style={{ maxWidth: '800px', width: '100%', margin: '0 auto' }}>
        <DetectionPanel 
          settings={settings} 
          onViolationsUpdate={handleViolationsUpdate} 
        />
      </div>
    </div>
  );
}
