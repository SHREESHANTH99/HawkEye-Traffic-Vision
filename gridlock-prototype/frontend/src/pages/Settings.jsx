import React from 'react';
import { useSettings } from '../contexts/SettingsContext';
import '../components/Sidebar.css';

export default function Settings() {
  const { settings, setSettings, handleClearLog } = useSettings();
  const [cleared, setCleared] = React.useState(false);

  const handleChange = (key, value) => {
    setSettings({ ...settings, [key]: value });
  };

  return (
    <div className="page-container" style={{ maxWidth: '600px', width: '100%', margin: '0 auto' }}>
      <h2 className="page-title">Configuration & Rules</h2>
      
      <div className="settings-panel">
        <div className="section-header">
          <span>Model Config</span>
        </div>
        
        <div className="control-group">
          <div className="slider-header">
            <label>Confidence Threshold</label>
            <span className="slider-value">{settings.confThreshold.toFixed(2)}</span>
          </div>
          <input 
            type="range" 
            className="custom-slider"
            min="0.1" max="0.9" step="0.05"
            value={settings.confThreshold}
            onChange={(e) => handleChange('confThreshold', parseFloat(e.target.value))}
            style={{ '--val': `${((settings.confThreshold - 0.1) / 0.8) * 100}%` }}
          />
        </div>

        <div className="control-group" style={{ marginBottom: 0 }}>
          <div className="slider-header">
            <label>Overlap Threshold</label>
            <span className="slider-value">{settings.overlapThreshold.toFixed(2)}</span>
          </div>
          <input 
            type="range" 
            className="custom-slider"
            min="0.1" max="0.8" step="0.05"
            value={settings.overlapThreshold}
            onChange={(e) => handleChange('overlapThreshold', parseFloat(e.target.value))}
            style={{ '--val': `${((settings.overlapThreshold - 0.1) / 0.7) * 100}%` }}
          />
        </div>
      </div>

      <div className="settings-panel">
        <div className="section-header">
          <span>Violation Rules</span>
        </div>
        
        <label className="custom-checkbox">
          <input 
            type="checkbox" 
            checked={settings.checkHelmet}
            onChange={(e) => handleChange('checkHelmet', e.target.checked)}
          />
          <span className="checkmark"></span>
          <span className="label-text">No Helmet Detection</span>
        </label>

        <label className="custom-checkbox">
          <input 
            type="checkbox" 
            checked={settings.checkTriple}
            onChange={(e) => handleChange('checkTriple', e.target.checked)}
          />
          <span className="checkmark"></span>
          <span className="label-text">Triple Riding Detection</span>
        </label>

        {settings.checkTriple && (
          <div className="control-group child-control">
            <label>Triple Riding Threshold</label>
            <div className="number-input-wrapper">
              <input 
                type="number" 
                min="2" max="5" 
                value={settings.tripleThreshold}
                onChange={(e) => handleChange('tripleThreshold', parseInt(e.target.value, 10))}
              />
              <span className="number-suffix">riders</span>
            </div>
          </div>
        )}

        <label className="custom-checkbox" style={{ marginBottom: 0 }}>
          <input 
            type="checkbox" 
            checked={settings.checkSignal}
            onChange={(e) => handleChange('checkSignal', e.target.checked)}
          />
          <span className="checkmark"></span>
          <span className="label-text">Signal Jump Detection</span>
        </label>
      </div>

      <div className="settings-panel">
        <div className="section-header">
          <span>Input Mode</span>
        </div>
        
        <div className="segmented-control">
          <button 
            className={settings.inputMode === 'image' ? 'active' : ''}
            onClick={() => handleChange('inputMode', 'image')}
          >
            Image
          </button>
          <button 
            className={settings.inputMode === 'video' ? 'active' : ''}
            onClick={() => handleChange('inputMode', 'video')}
          >
            Video
          </button>
        </div>
      </div>

      <div style={{ marginTop: '32px' }}>
        <button 
          className="btn-destructive w-full" 
          onClick={async () => {
            await handleClearLog();
            setCleared(true);
            setTimeout(() => setCleared(false), 2000);
          }}
          style={{ borderColor: cleared ? 'var(--semantic-success)' : '', color: cleared ? 'var(--semantic-success)' : '' }}
        >
          {cleared ? '✓ Server Logs Cleared' : 'Permanently Clear Server Log Data'}
        </button>
      </div>
    </div>
  );
}
