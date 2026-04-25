document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictionForm');
    const resultContainer = document.getElementById('resultContainer');
    const riskScore = document.getElementById('riskScore');
    const riskLevel = document.getElementById('riskLevel');
    const recommendationsList = document.getElementById('recommendationsList');
    const downloadBtn = document.getElementById('downloadReport');
    let xaiChart = null;
    let lastResult = null;

    async function generateSim(riskType) {
        const btn = event.target;
        const originalText = btn.innerText;
        btn.innerText = "Auditioning Candidates...";
        btn.disabled = true;

        try {
            const response = await fetch('/generate_sample', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ target: riskType })
            });
            const data = await response.json();
            
            if (data.status === 'error') throw new Error(data.message);
            
            await autoFillForm(data);
            
            // Auto-trigger analysis removed for manual presentation flow.
            
        } catch (error) {
            console.error("Simulation error:", error);
            alert("Simulation failed. Ensure Pulse AI is running.");
        } finally {
            btn.innerText = originalText;
            btn.disabled = false;
        }
    }

    async function autoFillForm(data) {
        const form = document.getElementById('predictionForm');
        
        // Use a promise to track completion
        return new Promise(resolve => {
            const entries = Object.entries(data);
            entries.forEach(([key, value], index) => {
                const field = form.elements[key];
                if (field && key !== 'target') {
                    setTimeout(() => {
                        field.value = value;
                        field.style.transition = 'background 0.3s';
                        field.style.background = 'rgba(0, 242, 254, 0.4)';
                        setTimeout(() => field.style.background = '', 400);
                        
                        // Final resolve
                        if (index === entries.length - 1) setTimeout(resolve, 300);
                    }, index * 50); 
                }
            });
        });
    }

    // Export to window
    window.generateSim = generateSim;

    // Authentication Logic
    const loginOverlay = document.getElementById('loginOverlay');
    const loginForm = document.getElementById('loginForm');
    const logoutBtn = document.getElementById('logoutBtn');
    const historySection = document.getElementById('history');

    // Check session on load - REMOVED for absolute security as requested.
    // Professional Portal will now prompt for login every time the page is refreshed.

    loginForm.addEventListener('submit', async (e) => {
        e.preventDefault();
        const username = document.getElementById('username').value;
        const password = document.getElementById('password').value;

        try {
            const response = await fetch('/login', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ username, password })
            });
            const res = await response.json();
            
            if (res.status === 'success') {
                localStorage.setItem('pulse_session', 'active');
                unlockApp();
            } else {
                alert(res.message);
            }
        } catch (err) {
            alert("Connection error.");
        }
    });

    logoutBtn.addEventListener('click', async () => {
        await fetch('/logout', { method: 'POST' });
        localStorage.removeItem('pulse_session');
        window.location.reload();
    });

    function unlockApp() {
        loginOverlay.classList.add('hidden');
        logoutBtn.classList.remove('hidden');
        historySection.classList.remove('hidden');
        renderHistory();
    }

    // History Logic
    function saveToHistory(data, result) {
        const history = JSON.parse(localStorage.getItem('pulse_history') || '[]');
        const record = {
            id: Date.now(),
            date: new Date().toLocaleString(),
            profile: `Age ${data.age}, ${data.sex === '1' ? 'Male' : 'Female'}`,
            probability: result.probability,
            level: result.recommendations.risk_level,
            fullData: { ...result, ...data } // Save for reload
        };
        history.unshift(record);
        localStorage.setItem('pulse_history', JSON.stringify(history.slice(0, 50))); // Keep last 50
        renderHistory();
    }

    function renderHistory() {
        const historyBody = document.getElementById('historyBody');
        const history = JSON.parse(localStorage.getItem('pulse_history') || '[]');
        
        historyBody.innerHTML = history.length ? '' : '<tr><td colspan="5" style="text-align:center">No clinical records found.</td></tr>';
        
        history.forEach(item => {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td>${item.date}</td>
                <td>${item.profile}</td>
                <td>${item.probability}%</td>
                <td><span class="risk-badge" style="background:${getRiskColor(item.probability)}; color:#000">${item.level}</span></td>
                <td><button class="btn-history-view" onclick="loadHistoryCase(${item.id})">View Case</button></td>
            `;
            historyBody.appendChild(tr);
        });
    }

    window.loadHistoryCase = (id) => {
        const history = JSON.parse(localStorage.getItem('pulse_history') || '[]');
        const record = history.find(r => r.id === id);
        if (record) {
            displayResults(record.fullData);
            resultContainer.classList.remove('hidden');
            resultContainer.scrollIntoView({ behavior: 'smooth' });
            lastResult = record.fullData;
        }
    };

    document.getElementById('clearHistory').addEventListener('click', () => {
        if (confirm("Permanently delete ALL patient history?")) {
            localStorage.removeItem('pulse_history');
            renderHistory();
        }
    });

    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        // Show loading state
        const submitBtn = form.querySelector('button');
        const originalText = submitBtn.innerText;
        submitBtn.innerText = 'Analyzing Cardiac Metrics...';
        submitBtn.disabled = true;

        const formData = new FormData(form);
        const data = Object.fromEntries(formData.entries());

        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });

            const result = await response.json();
            
            if (result.status === 'success') {
                lastResult = { ...result, ...data };
                displayResults(result);
                saveToHistory(data, result);
                resultContainer.classList.remove('hidden');
                resultContainer.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert('Error: ' + result.message);
            }
        } catch (err) {
            console.error(err);
            alert('Cardiac analysis failed. Please check your connection.');
        } finally {
            submitBtn.innerText = originalText;
            submitBtn.disabled = false;
        }
    });

    function displayResults(result) {
        // 1. Update Score & Progress
        const prob = result.probability;
        riskScore.innerText = `${prob}%`;
        
        const progress = document.querySelector('.circular-progress');
        progress.style.background = `conic-gradient(
            ${getRiskColor(prob)} ${prob * 3.6}deg, 
            rgba(255,255,255,0.05) 0deg
        )`;

        // 2. Risk Level Badge
        const recs = result.recommendations;
        riskLevel.innerText = `${recs.risk_level} Risk`;
        riskLevel.style.backgroundColor = getRiskColor(prob);
        riskLevel.style.color = '#000';

        // 3. Populate CDSS Sections
        populateList('dietEatList', recs.diet_to_eat);
        populateList('dietAvoidList', recs.diet_to_avoid);
        populateList('exerciseList', recs.lifestyle_exercises);
        
        document.getElementById('clinicalActionText').innerText = recs.clinical_action;
        document.getElementById('supplementsText').innerText = recs.supplements.join(', ');

        // 4. Render XAI Chart
        renderXAIChart(result.explanation);
    }

    function populateList(elementId, items) {
        const list = document.getElementById(elementId);
        list.innerHTML = '';
        items.forEach(item => {
            const li = document.createElement('li');
            li.innerHTML = item;
            list.appendChild(li);
        });
    }

    function renderXAIChart(explanation) {
        const ctx = document.getElementById('xaiChart').getContext('2d');
        
        if (xaiChart) xaiChart.destroy();

        // Sort by absolute impact
        const sorted = [...explanation].sort((a, b) => Math.abs(b.impact) - Math.abs(a.impact)).slice(0, 8);
        
        xaiChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: sorted.map(d => d.feature.toUpperCase()),
                datasets: [{
                    label: 'Feature Impact on Risk',
                    data: sorted.map(d => d.impact),
                    backgroundColor: sorted.map(d => d.impact > 0 ? 'rgba(239, 68, 68, 0.7)' : 'rgba(34, 197, 94, 0.7)'),
                    borderColor: sorted.map(d => d.impact > 0 ? '#ef4444' : '#22c55e'),
                    borderWidth: 1,
                    borderRadius: 5
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.1)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f8fafc' }
                    }
                }
            }
        });
    }

    function getRiskColor(prob) {
        if (prob < 35) return '#22c55e'; // Green (No / Low)
        if (prob < 65) return '#f59e0b'; // Amber (Moderate)
        return '#ef4444'; // Red (High)
    }

    downloadBtn.addEventListener('click', async () => {
        if (!lastResult) return;
        
        downloadBtn.innerText = 'Generating PDF...';
        
        try {
            const response = await fetch('/generate_report', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(lastResult)
            });
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `PulseAI_Patient_Report.pdf`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (err) {
            alert('Failed to generate report.');
        } finally {
            downloadBtn.innerText = 'Download Clinical Report (PDF)';
        }
    });
});
