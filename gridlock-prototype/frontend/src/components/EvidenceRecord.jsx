import React from 'react';
import './EvidenceRecord.css';

/**
 * Normalized record shape:
 * {
 *   id: string | number,
 *   type: string,          // 'NO_HELMET', 'TRIPLE_RIDING', 'SIGNAL_JUMP'
 *   plateText: string,     // 'UNKNOWN' or plate string
 *   plateValid: boolean,   // true/false
 *   confidence: number,    // 0-1 (optional)
 *   timestamp: string,     // Formatted or ISO string
 *   frameId: number,       // optional
 *   vehicleId: string,     // optional
 *   imageUrl: string       // optional
 * }
 */

export default function EvidenceRecord({ record, variant = 'compact' }) {
  const isCompact = variant === 'compact';

  const formatType = (type) => {
    if (!type) return 'UNKNOWN';
    return type.replace(/_/g, ' ');
  };

  const badgeClass = record.type === 'NO_HELMET'     ? 'badge-error'
                   : record.type === 'TRIPLE_RIDING'  ? 'badge-warning'
                   : record.type === 'SIGNAL_JUMP'    ? 'badge-warning'
                   : record.type === 'SEATBELT_CHECK' ? 'badge-warning'
                   : 'badge-neutral';

  const accentBorder = record.type === 'NO_HELMET'     ? '#ef4444'
                     : record.type === 'TRIPLE_RIDING'  ? '#f59e0b'
                     : record.type === 'SIGNAL_JUMP'    ? '#f59e0b'
                     : record.type === 'SEATBELT_CHECK' ? '#f59e0b'
                     : '#444';

  const formatTime = (ts) => {
    if (!ts) return '—';
    if (ts.includes(' ')) return ts.split(' ')[1]; // "YYYY-MM-DD HH:MM:SS" -> "HH:MM:SS"
    try {
      const d = new Date(ts);
      if (isNaN(d)) return ts;
      return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    } catch {
      return ts;
    }
  };

  /* ── COMPACT: single flat row, no card chrome ────────────────────────── */
  if (isCompact) {
    return (
      <div className="er-log-row" style={{ borderLeftColor: accentBorder }}>
        <span className={`er-type-badge ${badgeClass}`}>{formatType(record.type)}</span>
        <span className="er-log-plate mono">
          {record.plateText && record.plateText !== 'UNKNOWN' && record.plateText !== 'UNREADABLE'
            ? record.plateText
            : <span className="er-log-no-plate">— no plate —</span>}
        </span>
        <span className="er-log-time mono">{formatTime(record.timestamp)}</span>
        {record.frameId !== undefined && record.frameId !== null && (
          <span className="er-log-frame mono">f{record.frameId}</span>
        )}
      </div>
    );
  }

  /* ── FULL: card with image + metadata ────────────────────────────────── */
  return (
    <div className="evidence-record full">
      <div className="er-header">
        <span className={`er-type-badge ${badgeClass}`}>{formatType(record.type)}</span>
        {record.confidence !== undefined && (
          <span className="er-confidence mono">{(record.confidence * 100).toFixed(1)}%</span>
        )}
      </div>

      <div className="er-content">
        <div className="er-media">
          {record.imageUrl ? (
            <>
              <img
                src={record.imageUrl}
                alt={`${record.type} evidence`}
                className="er-image"
                loading="lazy"
                onError={(e) => {
                  e.target.style.display = 'none';
                  const ph = e.target.nextSibling;
                  if (ph) ph.style.display = 'flex';
                }}
              />
              <div className="er-image-placeholder mono" style={{ display: 'none' }}>NO IMAGE</div>
            </>
          ) : (
            <div className="er-image-placeholder mono">NO IMAGE</div>
          )}
        </div>

        <div className="er-metadata">
          <div className="er-meta-row">
            <span className="er-meta-label">PLATE</span>
            {record.plateText && record.plateText !== 'UNKNOWN' ? (
              <div className="plate-dossier-tag">
                <span className={`status-dot ${record.plateValid ? 'valid' : 'invalid'}`} />
                <span className={`mono ${record.plateValid ? '' : 'text-muted'}`}>{record.plateText}</span>
              </div>
            ) : (
              <span className="er-meta-value mono text-muted">UNKNOWN</span>
            )}
          </div>

          <div className="er-meta-row">
            <span className="er-meta-label">TIME</span>
            <span className="er-meta-value mono">{formatTime(record.timestamp)}</span>
          </div>

          {record.frameId !== undefined && record.frameId !== null && (
            <div className="er-meta-row">
              <span className="er-meta-label">FRAME</span>
              <span className="er-meta-value mono">{record.frameId}</span>
            </div>
          )}

          {record.vehicleId && (
            <div className="er-meta-row">
              <span className="er-meta-label">VEHICLE ID</span>
              <span className="er-meta-value mono">{record.vehicleId}</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
