import React, { useState, useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import { checkHealth } from '../api/client';
import { connectJudgeWebSocket } from '../api/judgeClient';
import './Sidebar.css';

export default function Sidebar() {
  const [apiOnline, setApiOnline] = useState(false);
  const [judgeOnline, setJudgeOnline] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false); // Placeholder

  const judgeWsRef = useRef(null);

  useEffect(() => {
    // Check Main API Health
    const pollApi = async () => {
      try {
        await checkHealth();
        setApiOnline(true);
      } catch (e) {
        setApiOnline(false);
      }
    };
    pollApi();
    const apiInterval = setInterval(pollApi, 5000);

    // Check Judge WS
    const conn = connectJudgeWebSocket(
      (statsUpdate, violationEvent) => {
        setJudgeOnline(true);
      },
      () => {
        setJudgeOnline(false);
      }
    );
    judgeWsRef.current = conn;

    const judgeStateCheck = setInterval(() => {
      if (conn.getState && conn.getState() === WebSocket.OPEN) {
        setJudgeOnline(true);
      } else {
        setJudgeOnline(false);
      }
    }, 5000);

    return () => {
      clearInterval(apiInterval);
      clearInterval(judgeStateCheck);
      conn.close();
    };
  }, []);

  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="brand-badge">
          <div className="logo-container">
            <div className="logo-pulse"></div>
            <div className="logo-text">HAWKEYE</div>
          </div>
          <div className="logo-subtext">TRAFFIC VISION</div>
        </div>
      </div>

      <nav className="sidebar-nav">
        <div className="nav-tier">
          <div className="tier-label">OPERATIONS</div>
          
          <NavLink 
            to="/" 
            className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
            end
          >
            <svg className="nav-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" />
              <path d="M3 9h18" />
              <path d="M9 21V9" />
            </svg>
            <span className="nav-label">Dashboard</span>
          </NavLink>
          
          <NavLink 
            to="/detect" 
            className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
          >
            <svg className="nav-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3l-2.5-3z" />
              <circle cx="12" cy="13" r="3" />
            </svg>
            <span className="nav-label">Detection Studio</span>
          </NavLink>

          <NavLink 
            to="/judge" 
            className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
          >
            <svg className="nav-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M4 22h14a2 2 0 0 0 2-2V7.5L14.5 2H6a2 2 0 0 0-2 2v4" />
              <path d="M14 2v6h6" />
              <path d="m3 12.5 3 3 7-7" />
            </svg>
            <span className="nav-label">Judge Feed</span>
          </NavLink>
        </div>

        <div className="nav-tier">
          <div className="tier-label">SYSTEM</div>

          <NavLink 
            to="/settings" 
            className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
          >
            <svg className="nav-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" />
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z" />
            </svg>
            <span className="nav-label">Settings</span>
          </NavLink>
        </div>
      </nav>
      
      <div className="sidebar-footer">
        <div className="status-panel">
          <div className="status-panel-header">SYSTEM STATUS</div>
          
          <div className="status-row">
            <span className="status-label">MAIN API</span>
            <div className={`status-indicator ${apiOnline ? 'online' : 'offline'}`}></div>
          </div>
          
          <div className="status-row">
            <span className="status-label">LLM JUDGE</span>
            <div className={`status-indicator ${judgeOnline ? 'online' : 'offline'}`}></div>
          </div>

          <div className="status-row">
            <span className="status-label">VISION MODEL</span>
            {/* TODO: Wire real model loaded status here. Using hollow placeholder for now. */}
            <div className="status-indicator placeholder"></div>
          </div>
        </div>
      </div>
    </div>
  );
}
