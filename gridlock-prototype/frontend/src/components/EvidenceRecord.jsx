import React from 'react';
import './EvidenceRecord.css';

export default function EvidenceRecord({ record, variant = 'compact' }) {
  const formatType = (type) => {
    if (!type) return 'UNKNOWN';
    return type.replace(/_/g, ' ');
  };

  const formatTime = (ts) => {
    if (!ts) return '—';
    if (ts.includes(' ')) return ts.split(' ')[1];
    try {
      const d = new Date(ts);
      if (isNaN(d)) return ts;
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return ts;
    }
  };

  const isFull = variant === 'full';

  return (
    <tr className="evidence-row">
      {isFull && (
        <td className="er-cell-image">
          {record.imageUrl ? (
            <img 
              src={record.imageUrl} 
              alt="Evidence" 
              className="er-thumb" 
              loading="lazy" 
              onClick={() => window.open(record.imageUrl, '_blank', 'noopener,noreferrer')}
              title="Click to view full image"
            />
          ) : (
            <div className="er-thumb-placeholder">NO IMG</div>
          )}
        </td>
      )}
      <td className="er-cell-type govt-badge status-confirmed">
        {formatType(record.type)}
      </td>
      <td className="er-cell-plate mono">
        {record.plateText && record.plateText !== 'UNKNOWN' && record.plateText !== 'UNREADABLE'
          ? record.plateText
          : <span className="text-secondary">—</span>}
      </td>
      {isFull && (
        <td className="er-cell-conf mono">
          {record.confidence !== undefined ? `${(record.confidence * 100).toFixed(1)}%` : '—'}
        </td>
      )}
      <td className="er-cell-time mono">
        {formatTime(record.timestamp)}
      </td>
      <td className="er-cell-frame mono text-secondary">
        {record.frameId !== undefined && record.frameId !== null ? `f${record.frameId}` : '—'}
      </td>
    </tr>
  );
}
