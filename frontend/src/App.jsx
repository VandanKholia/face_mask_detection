import { useState, useRef, useCallback } from 'react';
import './App.css';

const API_URL = 'https://face-mask-detection-backend.onrender.com/predict';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = useCallback((file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please upload a valid image file (JPG, PNG, etc.)');
      return;
    }
    setError(null);
    setResult(null);
    setSelectedFile(file);
    setPreviewUrl(URL.createObjectURL(file));
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    handleFile(file);
  }, [handleFile]);

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    setDragging(true);
  }, []);

  const handleDragLeave = useCallback(() => {
    setDragging(false);
  }, []);

  const handleFileSelect = (e) => {
    handleFile(e.target.files[0]);
  };

  const removeImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handlePredict = async () => {
    if (!selectedFile) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || 'Prediction failed');
      }

      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to the server. Make sure Flask is running.');
    } finally {
      setLoading(false);
    }
  };

  const confidencePercent = result ? (result.confidence * 100).toFixed(1) : 0;
  const isMask = result?.label_index === 1;

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
       
        <h1>Face Mask Detector</h1>
        <p>Upload a face image and our CNN model will detect whether the person is wearing a mask or not.</p>
      </header>

      {/* Main Content */}
      <div className="main-content">
        {/* Upload Card */}
        <div className="glass-card">
          <div className="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
              <polyline points="17 8 12 3 7 8" />
              <line x1="12" y1="3" x2="12" y2="15" />
            </svg>
            Upload Image
          </div>

          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            onChange={handleFileSelect}
            style={{ display: 'none' }}
            id="file-upload"
          />

          {!previewUrl ? (
            <div
              className={`upload-zone ${dragging ? 'dragging' : ''}`}
              onClick={() => fileInputRef.current?.click()}
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
            >
              <div className="upload-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                  <polyline points="17 8 12 3 7 8" />
                  <line x1="12" y1="3" x2="12" y2="15" />
                </svg>
              </div>
              <div className="upload-text">
                <h3>Drop your image here</h3>
                <p>or <span className="browse-link">browse files</span></p>
                <p style={{ marginTop: '0.5rem', fontSize: '0.7rem' }}>JPG, PNG, WEBP supported</p>
              </div>
            </div>
          ) : (
            <div className="upload-zone has-image">
              <div className="preview-container">
                <img src={previewUrl} alt="Preview" className="preview-image" />
                <button className="remove-btn" onClick={removeImage} title="Remove image">
                  ✕
                </button>
              </div>
            </div>
          )}

          <button
            className="predict-btn"
            onClick={handlePredict}
            disabled={!selectedFile || loading}
          >
            {loading ? (
              <>
                <div className="spinner"></div>
                Analyzing...
              </>
            ) : (
              <>
                <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="11" cy="11" r="8" />
                  <line x1="21" y1="21" x2="16.65" y2="16.65" />
                </svg>
                Detect Mask
              </>
            )}
          </button>

          {error && (
            <div className="error-banner" style={{ marginTop: '1rem' }}>
              <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="12" cy="12" r="10" />
                <line x1="12" y1="8" x2="12" y2="12" />
                <line x1="12" y1="16" x2="12.01" y2="16" />
              </svg>
              {error}
            </div>
          )}
        </div>

        {/* Result Card */}
        <div className="glass-card">
          <div className="card-title">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
            Results
          </div>

          {!result ? (
            <div className="result-placeholder">
              <div className="result-placeholder-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                  <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                  <circle cx="8.5" cy="8.5" r="1.5" />
                  <polyline points="21 15 16 10 5 21" />
                </svg>
              </div>
              <h3>No results yet</h3>
              <p>Upload an image and click "Detect Mask"</p>
            </div>
          ) : (
            <div className="result-display">
              {/* Status */}
              <div className={`result-status ${isMask ? 'with-mask' : 'without-mask'}`}>
                <div className="status-icon">
                  {isMask ? '✅' : '⚠️'}
                </div>
                <div className="status-text">
                  <h3>{result.label}</h3>
                  <p>{isMask ? 'Face mask detected in the image' : 'No face mask detected in the image'}</p>
                </div>
              </div>

              {/* Confidence */}
              <div className={`confidence-section ${isMask ? 'with-mask' : 'without-mask'}`}>
                <div className="confidence-header">
                  <span className="confidence-label">Confidence Score</span>
                  <span className={`confidence-value ${isMask ? 'high' : 'low'}`}>
                    {confidencePercent}%
                  </span>
                </div>
                <div className="confidence-bar-track">
                  <div
                    className={`confidence-bar-fill ${isMask ? 'high' : 'low'}`}
                    style={{ width: `${confidencePercent}%` }}
                  ></div>
                </div>
              </div>

              {/* Details */}
              <div className="details-grid">
                <div className="detail-item">
                  <div className="label">Prediction</div>
                  <div className="value">{result.label}</div>
                </div>
                <div className="detail-item">
                  <div className="label">Class Index</div>
                  <div className="value">{result.label_index}</div>
                </div>
                <div className="detail-item">
                  <div className="label">Model</div>
                  <div className="value">CNN</div>
                </div>
                <div className="detail-item">
                  <div className="label">Input Size</div>
                  <div className="value">128 × 128</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="footer">
        <p>Built with TensorFlow & React — Deep Learning Semester 6 Project</p>
      </footer>
    </div>
  );
}

export default App;
