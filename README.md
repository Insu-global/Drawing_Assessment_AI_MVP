# Drawing Assessment AI MVP

A full-stack web application for uploading and displaying images and PDF files.

## Features

- Upload images (JPEG, PNG, GIF) and PDF files
- View uploaded files in a gallery
- Modern, responsive UI
- File validation and error handling

## Setup

### Backend
```bash
cd backend
npm install
npm run dev
```
Server runs on http://localhost:3001

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on http://localhost:5173

## Usage

1. Open http://localhost:5173 in your browser
2. Use the "Upload" page to select and upload files
3. Use the "Display" page to view all uploaded files
4. Click "View" to open files in a new tab

## API Endpoints

- `POST /upload` - Upload a file
- `GET /files` - Get list of uploaded files
- `GET /uploads/:filename` - Serve uploaded files