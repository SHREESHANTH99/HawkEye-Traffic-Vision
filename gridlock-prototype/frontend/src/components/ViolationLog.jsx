import React, { useState } from 'react';
import { readPlate } from '../api/client';
import EvidenceRecord from './EvidenceRecord';
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
    <div className="violation-log-container">
      <div className="log-section">
        <div className="log-header">
          <div className="log-title govt-badge">Live Violation Log</div>
          <button className="btn-export" onClick={exportCSV} disabled={violations.length === 0}>
            EXPORT CSV
          </button>
        </div>
        
        <div className="table-wrapper">
          <table className="violation-table">
            <thead>
              <tr>
                <th>TYPE</th>
                <th>PLATE</th>
                <th>TIME</th>
                <th>FRAME</th>
              </tr>
            </thead>
            <tbody>
              {violations.length === 0 ? (
                <tr>
                  <td colSpan="4" className="empty-cell">No violations logged yet</td>
                </tr>
              ) : (
                violations.slice(0, 50).map((v, i) => {
                  const record = {
                    id: i,
                    type: v.violation_type,
                    plateText: v.plate_text,
                    plateValid: v.plate_valid,
                    confidence: v.confidence,
                    timestamp: v.timestamp,
                    frameId: v.frame_id,
                  };
                  return <EvidenceRecord key={i} record={record} variant="compact" />;
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <hr className="section-divider" />

      <div className="alpr-test-section">
        <div className="alpr-header">
          <div className="alpr-title govt-badge">ALPR Quick Test</div>
          <label className="btn-upload">
            SELECT CROP
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
                <span className="detail-label">RAW</span>
                <span className="detail-val mono">{alprResult.raw_text}</span>
              </div>
              <div className="detail-row">
                <span className="detail-label">PLATE</span>
                <div className="plate-dossier-tag">
                  <span className={`status-dot ${alprResult.plate_valid ? 'valid' : 'invalid'}`} title={alprResult.plate_valid ? 'Clean Read' : 'Partial Read'}></span>
                  <span className={alprResult.plate_valid ? 'mono highlight-val' : 'mono text-muted'}>{alprResult.plate_number}</span>
                </div>
              </div>
              <div className="detail-row">
                <span className="detail-label">CONFIDENCE</span>
                <span className="detail-val mono">{(alprResult.confidence * 100).toFixed(1)}%</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
