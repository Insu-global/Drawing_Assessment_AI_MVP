import { useState, useEffect } from 'react';
import axios from 'axios';

const DisplayPage = () => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const fetchFiles = async () => {
    try {
      setLoading(true);
      const response = await axios.get('http://localhost:5000/files');
      setFiles(response.data);
      setError('');
    } catch (err) {
      setError('Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFiles();
  }, []);

  const isImage = (filename) => {
    const ext = filename.toLowerCase().split('.').pop();
    return ['jpg', 'jpeg', 'png', 'gif'].includes(ext);
  };

  const isPDF = (filename) => {
    const ext = filename.toLowerCase().split('.').pop();
    return ext === 'pdf';
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleString();
  };

  if (loading) {
    return (
      <div className="display-page">
        <div className="loading">Loading files...</div>
      </div>
    );
  }

  return (
    <div className="display-page">
      <div className="display-container">
        <div className="display-header">
          <h2>Uploaded Files</h2>
          <button onClick={fetchFiles} className="refresh-btn">
            Refresh
          </button>
        </div>

        {error && <div className="error-message">{error}</div>}

        {files.length === 0 ? (
          <div className="no-files">
            <p>No files uploaded yet.</p>
            <p>Go to the Upload page to add some files!</p>
          </div>
        ) : (
          <div className="files-grid">
            {files.map((file) => (
              <div key={file.filename} className="file-card">
                <div className="file-preview">
                  {isImage(file.filename) ? (
                    <img
                      src={`http://localhost:5000${file.path}`}
                      alt={file.filename}
                      className="file-image"
                    />
                  ) : isPDF(file.filename) ? (
                    <div className="pdf-preview">
                      <div className="pdf-icon">📄</div>
                      <span>PDF Document</span>
                    </div>
                  ) : (
                    <div className="file-icon">📁</div>
                  )}
                </div>
                
                <div className="file-info">
                  <h3>{file.filename}</h3>
                  <p>Uploaded: {formatDate(file.uploadedAt)}</p>
                </div>

                <div className="file-actions">
                  <a
                    href={`http://localhost:5000${file.path}`}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="view-btn"
                  >
                    View
                  </a>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default DisplayPage; 