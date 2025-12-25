// server.js - Express server for Wiki Whatiz
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const PYTHON_API_URL = process.env.PYTHON_API_URL || 'http://localhost:8000';

// Middleware
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

// Proxy API requests to Python backend
app.post('/api/ask', async (req, res) => {
    try {
        const response = await fetch(`${PYTHON_API_URL}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(req.body)
        });
        const data = await response.json();
        res.json(data);
    } catch (error) {
        console.error('API Error:', error);
        res.status(500).json({ error: 'Failed to connect to RAG backend', details: error.message });
    }
});

app.get('/api/status', async (req, res) => {
    try {
        const response = await fetch(`${PYTHON_API_URL}/status`);
        const data = await response.json();
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: 'Backend not available', cuda_available: false });
    }
});

app.post('/api/warmup', async (req, res) => {
    try {
        const response = await fetch(`${PYTHON_API_URL}/warmup`, { method: 'POST' });
        const data = await response.json();
        res.json(data);
    } catch (error) {
        res.status(500).json({ error: 'Failed to warm up cache' });
    }
});

// Serve the main page
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// Catch-all for SPA
app.get('*', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, () => {
    console.log(`\n🚀 Wiki Whatiz running at http://localhost:${PORT}`);
    console.log(`📡 Python API expected at ${PYTHON_API_URL}`);
    console.log(`\nMake sure to start the Python backend first!`);
});
