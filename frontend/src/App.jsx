import { BrowserRouter as Router, Routes, Route, Link } from 'react-router-dom';
import UploadPage from './components/UploadPage';
import DisplayPage from './components/DisplayPage';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <nav className="navbar">
          <div className="nav-container">
            <h1>File Upload & Display</h1>
            <div className="nav-links">
              <Link to="/" className="nav-link">Upload</Link>
              <Link to="/display" className="nav-link">Display</Link>
            </div>
          </div>
        </nav>
        
        <main className="main-content">
          <Routes>
            <Route path="/" element={<UploadPage />} />
            <Route path="/display" element={<DisplayPage />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
}

export default App;
