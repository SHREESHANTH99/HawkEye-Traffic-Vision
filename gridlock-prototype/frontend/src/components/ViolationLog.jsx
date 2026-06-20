import React, { useState } from 'react';
import { readPlate } from '../api/client';
import './ViolationLog.css';

export default function ViolationLog({ violations, onClear }) {
  const [alprFile, setAlprFile] = useState(null);
  const [alprResult, setAlprResult] = useState(null);
  const [alprError, setAlprError] = useState(null);
  const [alprLoading, setAlprLoading] = useState(false);
  const [previewUrl, setPreviewUrl] = useState(null);

  const exportCSV = () => {
    if (violations.length === 0) return;
    
    const headers = ['Type', 'Plate', 'Confidence', 'Frame', 'Time'];
    const rows = violations.slice(0, 50).map(v => 
      [v.violation_type, v.plate_text, v.confidence.toFixed(2), v.frame_id, v.timestamp]
    );
    
    const csvContent = "data:text/csv;charset=utf-8," 
      + headers.join(",") + "\n" 
      + rows.map(e => e.join(",")).join("\n");
      
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `gridlock_violations_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleAlprUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    setAlprFile(file);
    setPreviewUrl(URL.createObjectURL(file));
    setAlprLoading(true);
    setAlprError(null);
    setAlprResult(null);
    
    try {
      const res = await readPlate(file);
      setAlprResult(res);
    } catch (err) {
      setAlprError(err.message || 'ALPR failed');
    } finally {
      setAlprLoading(false);
    }
  };

  return (
    <div className="violation-log-panel">
      <div className="log-header">
        <div className="log-title">Live Log</div>
        <button className="btn-export" onClick={exportCSV} disabled={violations.length === 0}>
          Export CSV
        </button>
      </div>
      
      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Type</th>
              <th>Plate</th>
              <th>Conf</th>
              <th>Frame</th>
              <th className="align-right">Time</th>
            </tr>
          </thead>
          <tbody>
            {violations.slice(0, 50).map((v, i) => {
              const badgeClass = v.violation_type === 'NO_HELMET' ? 'badge-error' 
                               : v.violation_type === 'TRIPLE_RIDING' ? 'badge-warning' 
                               : 'badge-neutral';
              return (
                <tr key={i}>
                  <td>
                    <span className={`badge ${badgeClass}`}>
                      {v.violation_type.replace('_', ' ')}
                    </span>
                  </td>
                  <td className="mono plate-cell">{v.plate_text}</td>
                  <td className="mono text-muted">{v.confidence.toFixed(2)}</td>
                  <td className="mono text-muted">{v.frame_id}</td>
                  <td className="mono time-col align-right">{v.timestamp.split(' ')[1]}</td>
                </tr>
              );
            })}
            {violations.length === 0 && (
              <tr>
                <td colSpan="5" className="empty-state">
                  <div className="empty-icon">✓</div>
                  <div>No violations logged yet</div>
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      <div className="alpr-test-section">
        <div className="alpr-header">
          <div className="alpr-title">ALPR Quick Test</div>
          <label className="btn-upload">
            Select Crop
            <input type="file" accept="image/*" onChange={handleAlprUpload} className="hidden-input" />
          </label>
        </div>
        
        {alprLoading && <div className="alpr-msg">Analyzing plate data...</div>}
        {alprError && <div className="alpr-msg error">{alprError}</div>}
        
        {previewUrl && alprResult && (
          <div className="alpr-result">
            <div className="alpr-img-wrapper">
               <img src={previewUrl} alt="Crop" className="alpr-preview" />
            </div>
            <div className="alpr-details">
              <div className="detail-row">
                <span className="detail-label">Raw</span>
                <span className="detail-val mono">{alprResult.raw_text}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Corrected</span>
                <span className="detail-val mono highlight-val">{alprResult.plate_number}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">Confidence</span>
                <span className="detail-val mono">{(alprResult.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
