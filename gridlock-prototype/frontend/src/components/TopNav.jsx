import React, { useState, useEffect, useRef } from 'react';
import { NavLink } from 'react-router-dom';
import { checkHealth } from '../api/client';
import { connectJudgeWebSocket } from '../api/judgeClient';
import './TopNav.css';

export default function TopNav() {
  const [apiOnline, setApiOnline] = useState(false);
  const [judgeOnline, setJudgeOnline] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);
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

  const closeMenu = () => setMenuOpen(false);

  return (
    <header className="topnav">
      <div className="topnav-container">
        <div className="topnav-brand">
          <div className="brand-logo">HAWKEYE</div>
          <div className="brand-subtext">Traffic Vision System</div>
        </div>

        {/* Hamburger Toggle */}
        <button 
          className={`hamburger ${menuOpen ? 'active' : ''}`} 
          onClick={() => setMenuOpen(!menuOpen)}
          aria-label="Toggle menu"
        >
          <span></span>
          <span></span>
          <span></span>
        </button>

        <nav className={`topnav-links ${menuOpen ? 'open' : ''}`}>
          <NavLink to="/" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"} onClick={closeMenu} end>Dashboard</NavLink>
          <NavLink to="/detect" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"} onClick={closeMenu}>Detection Studio</NavLink>
          <NavLink to="/judge" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"} onClick={closeMenu}>Judge Feed</NavLink>
          <NavLink to="/settings" className={({ isActive }) => isActive ? "nav-link active" : "nav-link"} onClick={closeMenu}>Settings</NavLink>
          
          {/* Status dots show up in menu on mobile */}
          <div className="mobile-status-container">
            <div className="status-item">
              <span className="status-label">MAIN API</span>
              <span className={`status-dot ${apiOnline ? 'valid' : 'invalid'}`}></span>
            </div>
            <div className="status-item">
              <span className="status-label">LLM JUDGE</span>
              <span className={`status-dot ${judgeOnline ? 'valid' : 'invalid'}`}></span>
            </div>
            <div className="status-item">
              <span className="status-label">VISION</span>
              <span className="status-dot placeholder"></span>
            </div>
          </div>
        </nav>

        <div className="topnav-status desktop-only">
          <div className="status-item">
            <span className="status-label">MAIN API</span>
            <span className={`status-dot ${apiOnline ? 'valid' : 'invalid'}`}></span>
          </div>
          <div className="status-item">
            <span className="status-label">LLM JUDGE</span>
            <span className={`status-dot ${judgeOnline ? 'valid' : 'invalid'}`}></span>
          </div>
          <div className="status-item">
            <span className="status-label">VISION MODEL</span>
            <span className="status-dot placeholder"></span>
          </div>
        </div>
      </div>
    </header>
  );
}
