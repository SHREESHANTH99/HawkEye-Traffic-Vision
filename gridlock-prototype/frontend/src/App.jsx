import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { SettingsProvider } from './contexts/SettingsContext';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import DetectionStudio from './pages/DetectionStudio';
import Settings from './pages/Settings';
import JudgeFeed from './pages/JudgeFeed';
import './App.css';

function App() {
  return (
    <SettingsProvider>
      <BrowserRouter>
        <div className="app-container">
          <Sidebar />
          
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/detect" element={<DetectionStudio />} />
              <Route path="/settings" element={<Settings />} />
              <Route path="/judge" element={<JudgeFeed />} />
            </Routes>
          </main>
        </div>
      </BrowserRouter>
    </SettingsProvider>
  );
}

export default App;
