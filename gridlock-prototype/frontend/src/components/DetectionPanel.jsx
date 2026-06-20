import React, { useState, useRef, useEffect } from 'react';
import { detectImage, streamVideoDetection } from '../api/client';
import './DetectionPanel.css';

export default function DetectionPanel({ settings, onViolationsUpdate }) {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [resultImage, setResultImage] = useState(null);
  const [currentViolations, setCurrentViolations] = useState([]);
  const [progress, setProgress] = useState(0);
  const [processingVideo, setProcessingVideo] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  
  const streamRef = useRef(null);

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.close();
      }
    };
  }, []);

  const handleFileChange = async (e) => {
    const selectedFile = e.target.files[0];
    if (!selectedFile) return;
    
    setFile(selectedFile);
    setError(null);
    setResultImage(null);
    setCurrentViolations([]);
    setProgress(0);

    if (settings.inputMode === 'image') {
      await processImage(selectedFile);
    } else if (settings.inputMode === 'video') {
      startVideoProcessing(selectedFile);
    }
  };

  const processImage = async (imageFile) => {
    setLoading(true);
    setStatusMessage('Uploading image for detection...');
    try {
      const data = await detectImage(imageFile, settings);
      setResultImage(data.annotated_image_b64);
      setCurrentViolations(data.violations || []);
      onViolationsUpdate(data.violations || []);
      setStatusMessage('Image processed successfully.');
    } catch (err) {
      setError(err.message);
      setStatusMessage('Image processing failed.');
    } finally {
      setLoading(false);
    }
  };

  const startVideoProcessing = (videoFile = file) => {
    if (!videoFile) return;
    setProcessingVideo(true);
    setError(null);
    setStatusMessage('Uploading video and starting detection...');
    setProgress(0);
    setCurrentViolations([]);
    
    streamRef.current = streamVideoDetection(
      videoFile,
      settings,
      (data) => {
        if (data.annotated_image_b64) {
          setResultImage(data.annotated_image_b64);
        }
        if (data.violations && data.violations.length > 0) {
          setCurrentViolations(data.violations);
          onViolationsUpdate(data.violations);
          setStatusMessage('Violations detected. Displaying latest annotated frame.');
        }
        if (data.progress !== undefined) {
          const pct = Math.round(data.progress * 100);
          setProgress(pct);
          setStatusMessage(`Processing video… ${pct}% complete`);
        }
      },
      () => {
        setProcessingVideo(false);
        setProgress(100);
        setStatusMessage('Video processing complete.');
        streamRef.current = null;
      },
      (err) => {
        setError(err.message || 'Video processing failed');
        setProcessingVideo(false);
        setStatusMessage('Video processing failed.');
        streamRef.current = null;
      }
    );
  };

  return (
    <div className="detection-panel">
      <div className="panel-header">
        <div className="panel-title">Detection Interface</div>
      </div>
      
      {!resultImage && !loading && !processingVideo && (
        <div className="upload-area">
          <label className="file-input-label">
            <div className="upload-icon">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <div className="upload-text">
              Upload {settings.inputMode === 'image' ? 'Image' : 'Video'}
            </div>
            <div className="upload-subtext">Click or drag and drop</div>
            <input 
              type="file" 
              accept={settings.inputMode === 'image' ? 'image/*' : 'video/mp4,video/x-m4v,video/*'} 
              onChange={handleFileChange}
            />
          </label>
        </div>
      )}

      {file && !resultImage && !loading && !processingVideo && (
        <div className="file-ready">
          <div className="file-name">{file.name}</div>
        </div>
      )}

      {settings.inputMode === 'video' && file && !processingVideo && progress === 0 && (
        <button className="btn-primary" onClick={startVideoProcessing}>
          Process Video Stream
        </button>
      )}

      {loading && (
        <div className="loading-state">
          <div className="spinner"></div>
          <div className="loading-text">Analyzing frame...</div>
        </div>
      )}

      {statusMessage && (
        <div className="status-box">
          <div className="status-text">{statusMessage}</div>
        </div>
      )}

      {processingVideo && (
        <div className="progress-state">
          <div className="progress-info">
            <span className="progress-label">Processing Stream</span>
            <span className="progress-pct">{Math.round(progress)}%</span>
          </div>
          <div className="progress-bar">
            <div className="progress-fill" style={{ width: `${progress}%` }}></div>
          </div>
        </div>
      )}

      {error && (
        <div className="error-box">
          <div className="error-icon">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
          </div>
          <div className="error-text">{error}</div>
        </div>
      )}

      {resultImage && (
        <div className="result-container">
          <img 
            src={`data:image/jpeg;base64,${resultImage}`} 
            alt="Annotated Result" 
            className="result-image" 
          />
          {settings.inputMode === 'image' && (
            <button className="btn-icon reset-btn" onClick={() => { setResultImage(null); setFile(null); }}>
               <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"></path><polyline points="9 22 9 12 15 12 15 22"></polyline></svg>
               <span>Upload Another</span>
            </button>
          )}
        </div>
      )}
    </div>
  );
}
