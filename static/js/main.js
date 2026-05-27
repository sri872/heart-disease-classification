document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const form = document.getElementById('predictionForm');
    const uploadZone = document.getElementById('uploadZone');
    const reportFileInput = document.getElementById('reportFile');
    const fileStatusBox = document.getElementById('fileStatusBox');
    const parsedParamsPanel = document.getElementById('parsedParamsPanel');
    const extractedRawText = document.getElementById('extractedRawText');

    let xaiChart = null;
    let lastResult = null;

    // --- PDF Drag-and-Drop & File Selection Handling ---
    uploadZone.addEventListener('click', () => reportFileInput.click());

    uploadZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadZone.classList.add('dragover');
    });

    uploadZone.addEventListener('dragleave', () => {
        uploadZone.classList.remove('dragover');
    });

    uploadZone.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleReportFile(e.dataTransfer.files[0]);
        }
    });

    reportFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleReportFile(e.target.files[0]);
        }
    });

    async function handleReportFile(file) {
        if (!file.name.toLowerCase().endsWith('.pdf')) {
            showFileStatus('Only PDF diagnostic reports are supported.', true);
            return;
        }

        showFileStatus(`<i class="fa-solid fa-spinner fa-spin"></i> Reading clinical parameters from <strong>${file.name}</strong>...`, false);
        
        const formData = new FormData();
        formData.append('report', file);

        try {
            const response = await fetch('/upload_report', {
                method: 'POST',
                body: formData
            });
            const data = await response.json();
            
            if (data.status === 'success') {
                showFileStatus(`<i class="fa-solid fa-circle-check"></i> Report <strong>${file.name}</strong> parsed successfully. Verify metrics below.`, false);
                displayParsedResults(data.features, data.metadata, data.raw_text);
            } else {
                showFileStatus(`<i class="fa-solid fa-circle-exclamation"></i> Parsing failed: ${data.message}`, true);
            }
        } catch (error) {
            console.error(error);
            showFileStatus('<i class="fa-solid fa-circle-exclamation"></i> Connection error reading PDF report.', true);
        }
    }

    function showFileStatus(htmlContent, isError) {
        fileStatusBox.innerHTML = htmlContent;
        fileStatusBox.className = `file-status-box ${isError ? 'error' : ''}`;
        fileStatusBox.classList.remove('hidden');
    }

    // --- Dynamic Population & Badge Management ---
    function displayParsedResults(features, metadata, rawText) {
        // Unhide verified metrics panel
        parsedParamsPanel.classList.remove('hidden');
        parsedParamsPanel.scrollIntoView({ behavior: 'smooth' });

        // Update form values
        for (const [key, val] of Object.entries(features)) {
            const field = document.getElementById(`input-${key}`);
            if (field) {
                field.value = val;
            }
            
            // Update parsed/default badge
            const badge = document.getElementById(`badge-${key}`);
            if (badge && metadata[key]) {
                const status = metadata[key].status;
                if (status === 'extracted') {
                    badge.className = 'parsed-badge extracted';
                    badge.innerText = 'Parsed';
                } else {
                    badge.className = 'parsed-badge default';
                    badge.innerText = 'Baseline';
                }
            }
        }

        // Extracted raw text logging
        extractedRawText.innerText = rawText || "No raw text log extracted.";
    }

    // --- Simulated Reports (Clinical Sandbox) ---
    async function simulateReport(riskType) {
        let mockData = {};
        let mockText = "";
        
        if (riskType === 'low') {
            mockText = "--- HOSPITAL INTRA-CLINICAL CASE SHEET ---\nPatient Name: John Doe\nAge: 42 Years Old\nGender: Female\nSubject exhibits no chest pain (asymptomatic under rest).\nResting Blood Pressure (BP): 115 mmHg\nSerum Cholesterol: 180 mg/dL\nFasting Blood Glucose: 95 mg/dL (Normal sugar levels)\nResting EKG: sinus rhythm (normal ecg)\nPeak Heart Rate (thalach): 168 bpm\nExercise-induced Angina: None (Negative)\nST Depression: 0.0 mm\nST Slope: upsloping under stress\nBlocked Vessels: 0 on fluoroscopy\nThalassemia: normal thal state\n-----------------------------------------";
            mockData = {
                features: { age: 42, sex: 0, cp: 3, trestbps: 115, chol: 180, fbs: 0, restecg: 0, thalach: 168, exang: 0, oldpeak: 0.0, slope: 0, ca: 0, thal: 1 },
                metadata: {
                    age: { status: 'extracted' }, sex: { status: 'extracted' }, cp: { status: 'extracted' },
                    trestbps: { status: 'extracted' }, chol: { status: 'extracted' }, fbs: { status: 'extracted' },
                    restecg: { status: 'extracted' }, thalach: { status: 'extracted' }, exang: { status: 'extracted' },
                    oldpeak: { status: 'extracted' }, slope: { status: 'extracted' }, ca: { status: 'extracted' }, thal: { status: 'extracted' }
                }
            };
        } else if (riskType === 'moderate') {
            mockText = "--- MOUNT SINAI CLINICAL ASSESSMENT ---\nCase Reference ID: CDSS-589-MOD\nPatient Demographics: 56 yrs of age\nSex / Gender: Male\nDiagnostic History: atypical chest discomfort observed\nCardiology Metrics:\nResting Blood Pressure: 135 mmHg\nTotal Cholesterol: 245 mg/dL\nFasting Blood Glucose level: 110 mg/dL\nResting ECG Segment: ST-T wave abnormality\nPeak Cardiac Performance: maximum heart rate of 142 bpm achieved\nExercise Induced Angina: positive yes\nST Segment Depression (oldpeak): 1.2 mm\nST Segment Slope: flat segment\nFluoroscopy blocked vessels (ca): 1 vessel\nThalassemia diagnosis: fixed defect\n---------------------------------------";
            mockData = {
                features: { age: 56, sex: 1, cp: 1, trestbps: 135, chol: 245, fbs: 0, restecg: 1, thalach: 142, exang: 1, oldpeak: 1.2, slope: 1, ca: 1, thal: 2 },
                metadata: {
                    age: { status: 'extracted' }, sex: { status: 'extracted' }, cp: { status: 'extracted' },
                    trestbps: { status: 'extracted' }, chol: { status: 'extracted' }, fbs: { status: 'extracted' },
                    restecg: { status: 'extracted' }, thalach: { status: 'extracted' }, exang: { status: 'extracted' },
                    oldpeak: { status: 'extracted' }, slope: { status: 'extracted' }, ca: { status: 'extracted' }, thal: { status: 'extracted' }
                }
            };
        } else if (riskType === 'high') {
            mockText = "--- CLINICAL OUTPATIENT CARDIOLOGY SUMMARY ---\nPATIENT INTAKE FORM\nAge: 68\nSex: Male\nPatient complaints: severe squeezing typical chest pain (typical angina)\nResting Blood Pressure: 155 mmHg\nSerum Cholesterol: 290 mg/dL\nFasting Blood Sugar (Glucose): 135 mg/dL\nResting EKG Assessment: LV Hypertrophy (lvh detected)\nMax Heart Rate Achieved (thalach): 118 bpm\nExercise Angina: yes present\nST Segment Depression (Oldpeak): 2.8 mm\nST Slope: downsloping\nMajor vessels blocked: 2 vessels colored\nThalassemia: reversible defect state\n----------------------------------------------";
            mockData = {
                features: { age: 68, sex: 1, cp: 0, trestbps: 155, chol: 290, fbs: 1, restecg: 2, thalach: 118, exang: 1, oldpeak: 2.8, slope: 2, ca: 2, thal: 3 },
                metadata: {
                    age: { status: 'extracted' }, sex: { status: 'extracted' }, cp: { status: 'extracted' },
                    trestbps: { status: 'extracted' }, chol: { status: 'extracted' }, fbs: { status: 'extracted' },
                    restecg: { status: 'extracted' }, thalach: { status: 'extracted' }, exang: { status: 'extracted' },
                    oldpeak: { status: 'extracted' }, slope: { status: 'extracted' }, ca: { status: 'extracted' }, thal: { status: 'extracted' }
                }
            };
        }

        showFileStatus(`<i class="fa-solid fa-robot"></i> Loaded simulated patient report context. Verify parameters below.`, false);
        displayParsedResults(mockData.features, mockData.metadata, mockText);
    }

    // Export simulation to global window
    window.simulateReport = simulateReport;

    // --- Authentication Logic ---
    const loginOverlay = document.getElementById('loginOverlay');
    const loginForm = document.getElementById('loginForm');
    const logoutBtn = document.getElementById('logoutBtn');
    const historySection = document.getElementById('history');

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

    // --- Patient History Storage (Secured In localStorage) ---
    function saveToHistory(data, result) {
        const history = JSON.parse(localStorage.getItem('pulse_history') || '[]');
        const record = {
            id: Date.now(),
            date: new Date().toLocaleString(),
            profile: `Age ${data.age}, ${data.sex === '1' ? 'Male' : 'Female'}`,
            probability: result.probability,
            level: result.recommendations.risk_level,
            fullData: { ...result, ...data }
        };
        history.unshift(record);
        localStorage.setItem('pulse_history', JSON.stringify(history.slice(0, 50)));
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
            sessionStorage.setItem('pulse_diagnostic_case', JSON.stringify(record.fullData));
            window.location.href = '/results';
        }
    };

    document.getElementById('clearHistory').addEventListener('click', () => {
        if (confirm("Permanently delete ALL patient history?")) {
            localStorage.removeItem('pulse_history');
            renderHistory();
        }
    });

    // --- Neural Model Prediction Dispatch ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();
        
        const submitBtn = form.querySelector('button');
        const originalText = submitBtn.innerText;
        submitBtn.innerText = 'Calculating Cardiac Metrics...';
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
                const combinedCase = { ...result, ...data };
                sessionStorage.setItem('pulse_diagnostic_case', JSON.stringify(combinedCase));
                saveToHistory(data, result);
                
                // Redirect immediately to the beautiful, dedicated results webpage
                window.location.href = '/results';
            } else {
                alert('Classification Error: ' + result.message);
            }
        } catch (err) {
            console.error(err);
            alert('Cardiac classification failed. Please verify local server connection.');
        } finally {
            submitBtn.innerText = originalText;
            submitBtn.disabled = false;
        }
    });

    function getRiskColor(prob) {
        if (prob < 35) return '#22c55e';
        if (prob < 65) return '#f59e0b';
        return '#ef4444';
    }
});
