// Wiki Whatiz - Frontend Application

// API base - empty string when served from same origin (FastAPI)
// Change to '/api' if using Node.js proxy
const API_BASE = '';

// State
let isLoading = false;

// DOM Elements - initialized after DOM loads
let questionInput, askButton, loadingSection, resultsSection;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    // Initialize DOM elements
    questionInput = document.getElementById('questionInput');
    askButton = document.getElementById('askButton');
    loadingSection = document.getElementById('loadingSection');
    resultsSection = document.getElementById('resultsSection');
    
    checkStatus();
    setupEventListeners();
    setupCursorGlow();
});

// Cursor glow effect with smooth motion blur
function setupCursorGlow() {
    const cursorGlow = document.getElementById('cursorGlow');
    
    // Current and target positions
    let currentX = 0;
    let currentY = 0;
    let targetX = 0;
    let targetY = 0;
    
    // Lerp factor - lower = slower, smoother motion (0.02-0.1 range)
    const lerp = 0.06;
    
    document.addEventListener('mousemove', (e) => {
        targetX = e.clientX;
        targetY = e.clientY;
    });
    
    // Animation loop for smooth interpolation
    function animate() {
        // Lerp towards target position
        currentX += (targetX - currentX) * lerp;
        currentY += (targetY - currentY) * lerp;
        
        cursorGlow.style.left = currentX + 'px';
        cursorGlow.style.top = currentY + 'px';
        
        requestAnimationFrame(animate);
    }
    
    animate();
}

function setupEventListeners() {
    // Enter key to submit
    questionInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !isLoading) {
            askQuestion();
        }
    });

    // Focus animation
    questionInput.addEventListener('focus', () => {
        questionInput.parentElement.classList.add('focused');
    });

    questionInput.addEventListener('blur', () => {
        questionInput.parentElement.classList.remove('focused');
    });
}

// Check backend status
async function checkStatus() {
    try {
        const response = await fetch(`${API_BASE}/status`);
        const data = await response.json();
        
        if (data.cuda_available || data.status === 'ok') {
            console.log('Backend connected:', data.gpu_name || 'CPU mode');
        } else {
            throw new Error('Backend not available');
        }
    } catch (error) {
        console.warn('Backend not available');
    }
}

// Set question from example (exposed globally for onclick)
window.setQuestion = function(text) {
    questionInput.value = text;
    questionInput.focus();
}

// Toggle settings panel (exposed globally for onclick)
window.toggleSettings = function() {
    const panel = document.getElementById('settingsPanel');
    panel.classList.toggle('open');
}

// Warm up cache (exposed globally for onclick)
window.warmupCache = async function() {
    const button = document.querySelector('.warmup-button');
    const originalText = button.innerHTML;
    
    button.disabled = true;
    button.innerHTML = '<span>*yawns* Loading...</span>';
    
    try {
        const response = await fetch(`${API_BASE}/warmup`, { method: 'POST' });
        const data = await response.json();
        
        button.innerHTML = `<span>Fine, I'm awake. (${data.time_seconds}s)</span>`;
        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = false;
        }, 2000);
    } catch (error) {
        button.innerHTML = '<span>Couldn\'t wake up 😴</span>';
        setTimeout(() => {
            button.innerHTML = originalText;
            button.disabled = false;
        }, 2000);
    }
}

// Main ask function (exposed globally for onclick)
window.askQuestion = async function() {
    const question = questionInput.value.trim();
    
    if (!question) {
        shakeInput();
        return;
    }
    
    if (isLoading) return;
    
    isLoading = true;
    showLoading();
    
    // Get settings
    const settings = {
        question: question,
        retriever_mode: document.getElementById('retrieverMode').value,
        use_rerank: document.getElementById('useRerank').checked,
        use_iterative: document.getElementById('useIterative').checked,
        rerank_k: parseInt(document.getElementById('rerankK').value),
        refine_n: parseInt(document.getElementById('refineN').value),
        sarcastic_mode: true,  // Always sarcastic, baby!
    };
    
    try {
        // Update loading steps
        updateLoadingStep(1);
        
        const response = await fetch(`${API_BASE}/ask`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(settings)
        });
        
        updateLoadingStep(2);
        
        if (!response.ok) {
            throw new Error(`API error: ${response.status}`);
        }
        
        const data = await response.json();
        
        updateLoadingStep(3);
        
        // Small delay for visual feedback
        await sleep(300);
        
        displayResults(data);
        
    } catch (error) {
        console.error('Error:', error);
        displayError(error.message);
    } finally {
        isLoading = false;
        hideLoading();
    }
}

function showLoading() {
    loadingSection.classList.add('active');
    resultsSection.classList.remove('active');
    askButton.disabled = true;
    
    // Reset steps
    document.querySelectorAll('.step').forEach(step => {
        step.classList.remove('active', 'done');
    });
}

function hideLoading() {
    loadingSection.classList.remove('active');
    askButton.disabled = false;
}

function updateLoadingStep(stepNum) {
    const steps = document.querySelectorAll('.step');
    
    steps.forEach((step, index) => {
        if (index + 1 < stepNum) {
            step.classList.remove('active');
            step.classList.add('done');
        } else if (index + 1 === stepNum) {
            step.classList.add('active');
            step.classList.remove('done');
        }
    });
    
    const loadingText = document.getElementById('loadingText');
    const messages = [
        'Ugh, fine... searching...',
        'Digging through Wikipedia for you...',
        'Deciding if your question deserves an answer...',
        'Crafting a response you probably won\'t appreciate...'
    ];
    loadingText.textContent = messages[stepNum - 1] || messages[0];
}

function displayResults(data) {
    resultsSection.classList.add('active');
    
    // Answer
    const answerContent = document.getElementById('answerContent');
    answerContent.innerHTML = formatAnswer(data.answer);
    
    // Timing
    const timingBadge = document.getElementById('timingBadge');
    timingBadge.textContent = `${data.total_time.toFixed(2)}s`;
    
    // Sources
    const sourcesGrid = document.getElementById('sourcesGrid');
    const sourceCount = document.getElementById('sourceCount');
    
    sourceCount.textContent = `${data.sources.length} sources`;
    sourcesGrid.innerHTML = data.sources.map((source, index) => `
        <div class="source-card">
            <div class="source-header">
                <span class="source-number">${index + 1}</span>
                <span class="source-meta">
                    chunk_id: ${source.chunk_id} | ${source.source}
                    ${source.rerank_score ? ` | score: ${source.rerank_score.toFixed(3)}` : ''}
                </span>
            </div>
            <div class="source-text">${escapeHtml(source.text)}</div>
        </div>
    `).join('');
    
    // Refined queries
    const refinedSection = document.getElementById('refinedSection');
    const refinedQueries = document.getElementById('refinedQueries');
    
    if (data.refined_queries && data.refined_queries.length > 0) {
        refinedSection.classList.add('active');
        refinedQueries.innerHTML = data.refined_queries.map(q => 
            `<span class="refined-chip">${escapeHtml(q)}</span>`
        ).join('');
    } else {
        refinedSection.classList.remove('active');
    }
    
    // Latency bars
    displayLatency(data.timings);
    
    // Setup scroll-triggered animations for source cards
    setupScrollAnimations();
    
    // Scroll to answer card after a brief delay for DOM to settle
    setTimeout(() => {
        const answerCard = document.querySelector('.answer-card');
        if (answerCard) {
            const cardRect = answerCard.getBoundingClientRect();
            const scrollTarget = window.scrollY + cardRect.top - 100; // 100px padding from top
            window.scrollTo({
                top: scrollTarget,
                behavior: 'smooth'
            });
        }
    }, 100);
}

function displayLatency(timings) {
    const latencyBars = document.getElementById('latencyBars');
    
    if (!timings || Object.keys(timings).length === 0) {
        latencyBars.innerHTML = '<p style="color: var(--text-muted);">No timing data available</p>';
        return;
    }
    
    const maxTime = Math.max(...Object.values(timings));
    
    latencyBars.innerHTML = Object.entries(timings).map(([key, value]) => {
        const percentage = (value / maxTime) * 100;
        return `
            <div class="latency-row">
                <span class="latency-label">${formatLatencyLabel(key)}</span>
                <div class="latency-bar-container">
                    <div class="latency-bar" style="--target-width: ${percentage}%">
                        ${value.toFixed(3)}s
                    </div>
                </div>
            </div>
        `;
    }).join('');
}

function formatLatencyLabel(key) {
    const labels = {
        'bm25_retrieve': 'BM25 Retrieve',
        'vector_retrieve': 'Vector Retrieve',
        'hybrid_retrieve': 'Hybrid Retrieve',
        'hybrid_retrieve_0': 'Initial Retrieval',
        'hybrid_retrieve_refined_total': 'Refined Retrieval',
        'query_refine': 'Query Refinement',
        'rerank': 'Re-ranking',
        'dedup_merge': 'Dedup & Merge',
        'ollama_llm': 'LLM Generation',
    };
    return labels[key] || key;
}

function displayError(message) {
    resultsSection.classList.add('active');
    
    const answerContent = document.getElementById('answerContent');
    answerContent.innerHTML = `
        <div style="color: var(--error); padding: 1rem; background: rgba(239, 68, 68, 0.1); border-radius: 8px;">
            <strong>Well, this is embarrassing...</strong> ${escapeHtml(message)}
            <br><br>
            <small>The Python API backend isn't responding. Maybe it's on a coffee break? Start it on port 8000.</small>
        </div>
    `;
    
    document.getElementById('sourcesGrid').innerHTML = '';
    document.getElementById('sourceCount').textContent = '0 sources';
    document.getElementById('refinedSection').classList.remove('active');
    document.getElementById('latencyBars').innerHTML = '';
}

function formatAnswer(text) {
    // Convert [[important text]] to highlighted text using placeholders
    // The double brackets are easier for the LLM to produce correctly
    let html = text.replace(/\[\[([^\]]+)\]\]/g, '{{HIGHLIGHT}}$1{{/HIGHLIGHT}}');
    
    // Also handle **text** as fallback (in case LLM uses asterisks)
    html = html.replace(/\*\*([^*]+)\*\*/g, '{{HIGHLIGHT}}$1{{/HIGHLIGHT}}');
    
    // Clean up any leftover asterisks (edge cases)
    html = html.replace(/\*{2,}/g, '');
    
    // Now escape HTML for safety
    html = escapeHtml(html);
    
    // Convert our placeholders back to actual HTML
    html = html.replace(/\{\{HIGHLIGHT\}\}/g, '<span class="highlight">');
    html = html.replace(/\{\{\/HIGHLIGHT\}\}/g, '</span>');
    
    // Convert bullet points (- or *)
    html = html.replace(/^[-•]\s+(.+)$/gm, '<li>$1</li>');
    
    // Wrap consecutive li elements in ul
    html = html.replace(/((?:<li>.*<\/li>\s*)+)/g, '<ul>$1</ul>');
    
    // Convert paragraphs (double newlines)
    html = html.split(/\n\n+/).map(p => {
        p = p.trim();
        if (!p || p.startsWith('<ul>')) return p;
        return `<p>${p}</p>`;
    }).join('');
    
    // Convert single line breaks within paragraphs
    html = html.replace(/([^>])\n([^<])/g, '$1<br>$2');
    
    // Add staggered animation delay to highlights
    let highlightIndex = 0;
    html = html.replace(/class="highlight"/g, () => {
        return `class="highlight" style="animation-delay: ${highlightIndex++ * 0.2}s"`;
    });
    
    return html;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function shakeInput() {
    const wrapper = questionInput.parentElement;
    wrapper.style.animation = 'shake 0.5s ease';
    setTimeout(() => {
        wrapper.style.animation = '';
    }, 500);
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Add shake animation
const style = document.createElement('style');
style.textContent = `
    @keyframes shake {
        0%, 100% { transform: translateX(0); }
        25% { transform: translateX(-10px); }
        75% { transform: translateX(10px); }
    }
`;
document.head.appendChild(style);

// Intersection Observer for scroll-triggered animations
function setupScrollAnimations() {
    const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry, index) => {
            if (entry.isIntersecting) {
                // Add staggered delay based on element index within its parent
                const parent = entry.target.parentElement;
                const siblings = Array.from(parent.children);
                const idx = siblings.indexOf(entry.target);
                entry.target.style.animationDelay = `${idx * 0.1}s`;
                entry.target.classList.add('animate-in');
                observer.unobserve(entry.target); // Only animate once
            }
        });
    }, {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    });

    // Observe all source cards
    document.querySelectorAll('.source-card').forEach(card => {
        observer.observe(card);
    });
    
    // Observe all latency rows
    document.querySelectorAll('.latency-row').forEach(row => {
        observer.observe(row);
    });
}

// Call after displaying results
window.setupScrollAnimations = setupScrollAnimations;
