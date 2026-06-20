import React from 'react';
import { NavLink } from 'react-router-dom';
import './Sidebar.css';

export default function Sidebar() {
  return (
    <div className="sidebar">
      <div className="sidebar-header">
        <div className="logo-container">
          <div className="logo-pulse"></div>
          <div className="logo-text">GRIDLOCK</div>
        </div>
        <div className="logo-subtext">VIOLATION DETECTION</div>
      </div>

      <nav className="sidebar-nav">
        <NavLink 
          to="/" 
          className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
          end
        >
          <span className="nav-icon">📊</span>
          <span className="nav-label">Dashboard</span>
        </NavLink>
        
        <NavLink 
          to="/detect" 
          className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
        >
          <span className="nav-icon">📷</span>
          <span className="nav-label">Detection Studio</span>
        </NavLink>

        <NavLink 
          to="/settings" 
          className={({ isActive }) => isActive ? "nav-item active" : "nav-item"}
        >
          <span className="nav-icon">⚙️</span>
          <span className="nav-label">Settings</span>
        </NavLink>
      </nav>
      
      <div className="sidebar-footer" style={{ marginTop: 'auto' }}>
        <div className="footer-status">
          <div className="status-dot"></div>
          <span>System Online</span>
        </div>
      </div>
    </div>
  );
}
