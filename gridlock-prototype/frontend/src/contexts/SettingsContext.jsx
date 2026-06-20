import React, { createContext, useState, useContext, useEffect } from 'react';
import { fetchViolationSummary, fetchViolationLog, clearViolationLog } from '../api/client';

export const SettingsContext = createContext(null);

export function SettingsProvider({ children }) {
  const [settings, setSettings] = useState({
    confThreshold: 0.45,
    overlapThreshold: 0.30,
    tripleThreshold: 3,
    checkHelmet: true,
    checkTriple: true,
    checkSignal: false,
    inputMode: 'image',
  });

  const [summary, setSummary] = useState({
    total: 0,
    NO_HELMET: 0,
    TRIPLE_RIDING: 0,
    SIGNAL_JUMP: 0
  });
  
  const [violations, setViolations] = useState([]);
  const [framesProcessed, setFramesProcessed] = useState(0);

  useEffect(() => {
    const poll = async () => {
      try {
        const sumData = await fetchViolationSummary();
        setSummary(sumData);
        
        const logData = await fetchViolationLog(50);
        setViolations(logData.records || []);
      } catch (err) {
        console.error('Polling error:', err);
      }
    };
    
    poll();
    const interval = setInterval(poll, 3000);
    return () => clearInterval(interval);
  }, []);

  const handleClearLog = async () => {
    try {
      await clearViolationLog();
    } catch (e) {
      console.error(e);
    }
    setViolations([]);
    setSummary({
      total: 0,
      NO_HELMET: 0,
      TRIPLE_RIDING: 0,
      SIGNAL_JUMP: 0
    });
    setFramesProcessed(0);
  };

  const handleViolationsUpdate = (newViolations) => {
    setFramesProcessed(prev => prev + 1);
  };

  return (
    <SettingsContext.Provider value={{
      settings,
      setSettings,
      summary,
      violations,
      framesProcessed,
      handleClearLog,
      handleViolationsUpdate
    }}>
      {children}
    </SettingsContext.Provider>
  );
}

export function useSettings() {
  return useContext(SettingsContext);
}
