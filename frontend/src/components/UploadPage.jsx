import { useState } from 'react';
import axios from 'axios';

const UploadPage = () => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'application/pdf'];
      const maxSize = 100 * 1024 * 1024; // 100MB
      
      if (allowedTypes.includes(selectedFile.type)) {
        if (selectedFile.size <= maxSize) {
          setFile(selectedFile);
          setMessage('');
        } else {
          setMessage('File size must be less than 100MB.');
          setFile(null);
        }
      } else {
        setMessage('Please select a valid image or PDF file.');
        setFile(null);
      }
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setMessage('Please select a file first.');
      return;
    }

    setUploading(true);
    setMessage('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axios.post('http://localhost:3001/upload', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      setMessage('File uploaded successfully!');
      setFile(null);
      document.getElementById('file-input').value = '';
    } catch (error) {
      setMessage(error.response?.data?.error || 'Upload failed. Please try again.');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-page">
      <div className="upload-container">
        <h2>Upload File</h2>
        <p>Select an image or PDF file to upload (max 100MB)</p>
        
        <div className="file-input-container">
          <input
            type="file"
            id="file-input"
            accept="image/*,.pdf"
            onChange={handleFileChange}
            className="file-input"
          />
          <label htmlFor="file-input" className="file-label">
            Choose File
          </label>
        </div>

        {file && (
          <div className="file-info">
            <p>Selected: {file.name}</p>
            <p>Size: {(file.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        )}

        <button
          onClick={handleUpload}
          disabled={!file || uploading}
          className="upload-btn"
        >
          {uploading ? 'Uploading...' : 'Upload File'}
        </button>

        {message && (
          <div className={`message ${message.includes('successfully') ? 'success' : 'error'}`}>
            {message}
          </div>
        )}
      </div>
    </div>
  );
};

export default UploadPage; 