import os
import json
import joblib
import pandas as pd
import numpy as np
import shap
from flask import Flask, request, jsonify, render_template, send_file, session
from io import BytesIO
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

app = Flask(__name__)
app.secret_key = 'pulse_ai_secure_2026_clinical_key' # For session encryption

# Load Model, Scaler and Feature Names
model = joblib.load('pulse_ai_model.pkl')
scaler = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')

# Initialize SHAP Explainer
# We'll use a representative sample for the kernel explainer or use TreeExplainer for Random Forest
explainer = shap.TreeExplainer(model)

import re
from pypdf import PdfReader

def parse_pdf_report(pdf_file):
    # Extract text from PDF
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
            
    # Clean the text a bit for easier regex matching
    cleaned_text = re.sub(r'\s+', ' ', text)
    
    # Feature defaults (clinical averages)
    defaults = {
        'age': 54.0,
        'sex': 1.0,      # Male
        'cp': 0.0,       # Typical Angina
        'trestbps': 130.0,
        'chol': 240.0,
        'fbs': 0.0,      # Normal
        'restecg': 0.0,  # Normal
        'thalach': 150.0,
        'exang': 0.0,    # No
        'oldpeak': 1.0,
        'slope': 1.0,    # Flat
        'ca': 0.0,       # 0 vessels
        'thal': 2.0      # Fixed defect
    }
    
    # Descriptions for user interface
    descriptions = {
        'age': 'Age of the patient',
        'sex': 'Biological Sex (1=Male, 0=Female)',
        'cp': 'Chest Pain Type (0=Typical Angina, 1=Atypical Angina, 2=Non-Anginal, 3=Asymptomatic)',
        'trestbps': 'Resting Blood Pressure (mmHg)',
        'chol': 'Serum Cholesterol (mg/dL)',
        'fbs': 'Fasting Blood Sugar > 120 mg/dL (1=True, 0=False)',
        'restecg': 'Resting ECG Results (0=Normal, 1=ST-T Abnormality, 2=LV Hypertrophy)',
        'thalach': 'Maximum Heart Rate Achieved',
        'exang': 'Exercise Induced Angina (1=Yes, 0=No)',
        'oldpeak': 'ST Depression (Oldpeak)',
        'slope': 'ST Slope (0=Upsloping, 1=Flat, 2=Downsloping)',
        'ca': 'Major Blocked Vessels (0-3)',
        'thal': 'Thalassemia Type (1=Normal, 2=Fixed Defect, 3=Reversible Defect)'
    }
    
    parsed = {}
    metadata = {}
    
    # 1. Parse AGE
    age_match = re.search(r'(?i)\b(?:age|yr|yrs|years?)\s*[:\-]?\s*(\d{1,3})\b', cleaned_text)
    if age_match:
        parsed['age'] = float(age_match.group(1))
        metadata['age'] = {'status': 'extracted', 'val': parsed['age'], 'match': age_match.group(0)}
    else:
        # Check for year of birth
        dob_match = re.search(r'(?i)\b(?:dob|birth|born)\s*[:\-]?\s*\d{1,2}[/\-]\d{1,2}[/\-](\d{4})\b', cleaned_text)
        if not dob_match:
            dob_match = re.search(r'(?i)\b(?:dob|birth|born)\s*[:\-]?\s*(\d{4})\b', cleaned_text)
        if dob_match:
            birth_year = int(dob_match.group(1))
            calculated_age = 2026 - birth_year
            parsed['age'] = float(calculated_age)
            metadata['age'] = {'status': 'extracted', 'val': parsed['age'], 'match': dob_match.group(0)}
        else:
            parsed['age'] = defaults['age']
            metadata['age'] = {'status': 'default', 'val': defaults['age'], 'match': None}
            
    # 2. Parse SEX
    sex_match = re.search(r'(?i)\b(?:sex|gender)\s*[:\-]?\s*(male|female|m|f)\b', cleaned_text)
    if sex_match:
        val_str = sex_match.group(1).lower()
        if val_str in ['male', 'm']:
            parsed['sex'] = 1.0
        else:
            parsed['sex'] = 0.0
        metadata['sex'] = {'status': 'extracted', 'val': parsed['sex'], 'match': sex_match.group(0)}
    else:
        parsed['sex'] = defaults['sex']
        metadata['sex'] = {'status': 'default', 'val': defaults['sex'], 'match': None}
        
    # 3. Parse RESTING BLOOD PRESSURE (trestbps)
    bp_match = re.search(r'(?i)\b(?:bp|blood\s+pressure|resting\s+bp|sys|systolic|pressure)\s*[:\-]?\s*(\d{2,3})(?:\s*/\s*\d{2,3})?\b', cleaned_text)
    if bp_match:
        parsed['trestbps'] = float(bp_match.group(1))
        metadata['trestbps'] = {'status': 'extracted', 'val': parsed['trestbps'], 'match': bp_match.group(0)}
    else:
        parsed['trestbps'] = defaults['trestbps']
        metadata['trestbps'] = {'status': 'default', 'val': defaults['trestbps'], 'match': None}
        
    # 4. Parse CHOLESTEROL (chol)
    chol_match = re.search(r'(?i)\b(?:chol(?:esterol)?|total\s+chol(?:esterol)?|ldl|hdl|lipid|lipids)\s*[:\-]?\s*(\d{2,3})\b', cleaned_text)
    if chol_match:
        parsed['chol'] = float(chol_match.group(1))
        metadata['chol'] = {'status': 'extracted', 'val': parsed['chol'], 'match': chol_match.group(0)}
    else:
        parsed['chol'] = defaults['chol']
        metadata['chol'] = {'status': 'default', 'val': defaults['chol'], 'match': None}
        
    # 5. Parse FASTING BLOOD SUGAR (fbs)
    fbs_match = re.search(r'(?i)\b(?:fbs|fasting\s+blood\s+sugar|glucose|blood\s+sugar|sugar)\s*[:\-]?\s*(\d{2,3})\b', cleaned_text)
    if fbs_match:
        val = float(fbs_match.group(1))
        parsed['fbs'] = 1.0 if val > 120 else 0.0
        metadata['fbs'] = {'status': 'extracted', 'val': parsed['fbs'], 'match': fbs_match.group(0) + f" (Glucose: {val})"}
    else:
        if re.search(r'(?i)\b(?:diabetes|hyperglycemia)\b', cleaned_text):
            parsed['fbs'] = 1.0
            metadata['fbs'] = {'status': 'extracted', 'val': 1.0, 'match': 'Diabetes mention'}
        else:
            parsed['fbs'] = defaults['fbs']
            metadata['fbs'] = {'status': 'default', 'val': defaults['fbs'], 'match': None}
            
    # 6. Parse CHEST PAIN TYPE (cp)
    cp_matched = False
    for cp_type, code, keywords in [
        ('typical', 0.0, [r'typical\s+angina', r'typical\s+chest\s+pain', r'angina\s+pectoris']),
        ('atypical', 1.0, [r'atypical\s+angina', r'atypical\s+chest\s+pain', r'atypical\s+chest\s+discomfort']),
        ('non-anginal', 2.0, [r'non\-anginal', r'non\-cardiac\s+chest\s+pain']),
        ('asymptomatic', 3.0, [r'asymptomatic', r'no\s+chest\s+pain', r'no\s+angina'])
    ]:
        for kw in keywords:
            m = re.search(r'(?i)\b' + kw + r'\b', cleaned_text)
            if m:
                parsed['cp'] = code
                metadata['cp'] = {'status': 'extracted', 'val': code, 'match': m.group(0)}
                cp_matched = True
                break
        if cp_matched:
            break
            
    if not cp_matched:
        generic_cp = re.search(r'(?i)\b(?:chest\s+pain|angina)\b', cleaned_text)
        if generic_cp:
            parsed['cp'] = 0.0
            metadata['cp'] = {'status': 'extracted', 'val': 0.0, 'match': generic_cp.group(0)}
        else:
            parsed['cp'] = defaults['cp']
            metadata['cp'] = {'status': 'default', 'val': defaults['cp'], 'match': None}
            
    # 7. Parse RESTING ECG (restecg)
    ecg_matched = False
    for ecg_type, code, keywords in [
        ('st-t', 1.0, [r'st\-t\s+abnormality', r'st\s+elevation', r'st\s+depression', r't\s+wave\s+inversion']),
        ('lvh', 2.0, [r'left\s+ventricular\s+hypertrophy', r'lv\s+hypertrophy', r'lvh']),
        ('normal', 0.0, [r'normal\s+ecg', r'normal\s+ekg', r'sinus\s+rhythm'])
    ]:
        for kw in keywords:
            m = re.search(r'(?i)\b' + kw + r'\b', cleaned_text)
            if m:
                parsed['restecg'] = code
                metadata['restecg'] = {'status': 'extracted', 'val': code, 'match': m.group(0)}
                ecg_matched = True
                break
        if ecg_matched:
            break
            
    if not ecg_matched:
        parsed['restecg'] = defaults['restecg']
        metadata['restecg'] = {'status': 'default', 'val': defaults['restecg'], 'match': None}
        
    # 8. Parse MAX HEART RATE (thalach)
    hr_match = re.search(r'(?i)\b(?:max(?:imum)?\s+(?:heart\s+rate|hr)|peak\s+hr|thalach|heart\s+rate|pulse)\s*[:\-]?\s*(\d{2,3})\b', cleaned_text)
    if hr_match:
        parsed['thalach'] = float(hr_match.group(1))
        metadata['thalach'] = {'status': 'extracted', 'val': parsed['thalach'], 'match': hr_match.group(0)}
    else:
        parsed['thalach'] = defaults['thalach']
        metadata['thalach'] = {'status': 'default', 'val': defaults['thalach'], 'match': None}
        
    # 9. Parse EXERCISE INDUCED ANGINA (exang)
    exang_match = re.search(r'(?i)\b(?:exercise\s+induced\s+angina|exang|angina\s+on\s+exertion|angina\s+with\s+exercise)\s*[:\-]?\s*(yes|no|positive|negative|1|0)\b', cleaned_text)
    if exang_match:
        val_str = exang_match.group(1).lower()
        if val_str in ['yes', 'positive', '1']:
            parsed['exang'] = 1.0
        else:
            parsed['exang'] = 0.0
        metadata['exang'] = {'status': 'extracted', 'val': parsed['exang'], 'match': exang_match.group(0)}
    else:
        parsed['exang'] = defaults['exang']
        metadata['exang'] = {'status': 'default', 'val': defaults['exang'], 'match': None}
        
    # 10. Parse OLDPEAK
    oldpeak_match = re.search(r'(?i)\b(?:st\s+depression|oldpeak|st\s+dep)\s*[:\-]?\s*(\d+(?:\.\d+)?)\b', cleaned_text)
    if oldpeak_match:
        parsed['oldpeak'] = float(oldpeak_match.group(1))
        metadata['oldpeak'] = {'status': 'extracted', 'val': parsed['oldpeak'], 'match': oldpeak_match.group(0)}
    else:
        parsed['oldpeak'] = defaults['oldpeak']
        metadata['oldpeak'] = {'status': 'default', 'val': defaults['oldpeak'], 'match': None}
        
    # 11. Parse SLOPE
    slope_matched = False
    for slope_type, code, keywords in [
        ('upsloping', 0.0, [r'upsloping', r'up\-sloping']),
        ('flat', 1.0, [r'flat', r'horizontal']),
        ('downsloping', 2.0, [r'downsloping', r'down\-sloping'])
    ]:
        for kw in keywords:
            m = re.search(r'(?i)\b' + kw + r'\b', cleaned_text)
            if m:
                parsed['slope'] = code
                metadata['slope'] = {'status': 'extracted', 'val': code, 'match': m.group(0)}
                slope_matched = True
                break
        if slope_matched:
            break
            
    if not slope_matched:
        parsed['slope'] = defaults['slope']
        metadata['slope'] = {'status': 'default', 'val': defaults['slope'], 'match': None}
        
    # 12. Parse MAJOR VESSELS (ca)
    ca_match = re.search(r'(?i)\b(?:fluoroscopy|vessels|ca|blocked)\s*[:\-]?\s*([0-3])\b', cleaned_text)
    if ca_match:
        parsed['ca'] = float(ca_match.group(1))
        metadata['ca'] = {'status': 'extracted', 'val': parsed['ca'], 'match': ca_match.group(0)}
    else:
        parsed['ca'] = defaults['ca']
        metadata['ca'] = {'status': 'default', 'val': defaults['ca'], 'match': None}
        
    # 13. Parse THALASSEMIA (thal)
    thal_matched = False
    for thal_type, code, keywords in [
        ('normal', 1.0, [r'normal\s+thal', r'no\s+thalassemia']),
        ('fixed', 2.0, [r'fixed\s+defect', r'fixed\s+thal']),
        ('reversible', 3.0, [r'reversible\s+defect', r'reversible\s+thal'])
    ]:
        for kw in keywords:
            m = re.search(r'(?i)\b' + kw + r'\b', cleaned_text)
            if m:
                parsed['thal'] = code
                metadata['thal'] = {'status': 'extracted', 'val': code, 'match': m.group(0)}
                thal_matched = True
                break
        if thal_matched:
            break
            
    if not thal_matched:
        parsed['thal'] = defaults['thal']
        metadata['thal'] = {'status': 'default', 'val': defaults['thal'], 'match': None}
        
    return parsed, metadata, text

@app.route('/upload_report', methods=['POST'])
def upload_report():
    try:
        if 'report' not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"}), 400
            
        file = request.files['report']
        if file.filename == '':
            return jsonify({"status": "error", "message": "Empty file name"}), 400
            
        if not file.filename.lower().endswith('.pdf'):
            return jsonify({"status": "error", "message": "Only PDF diagnostic reports are supported"}), 400
            
        parsed, metadata, raw_text = parse_pdf_report(file)
        
        return jsonify({
            "status": "success",
            "features": parsed,
            "metadata": metadata,
            "raw_text": raw_text
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/login', methods=['POST'])
def login():
    data = request.json
    # Professional demo credentials
    if data.get('username') == 'physician_01' and data.get('password') == 'pulse2026':
        session['user'] = 'physician_01'
        return jsonify({"status": "success", "message": "Access Granted"})
    return jsonify({"status": "error", "message": "Invalid Clinical Credentials"}), 401

@app.route('/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({"status": "success"})

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        # Extract features in the correct order
        input_data = [float(data.get(feat, 0)) for feat in feature_names]
        
        # Scale the data
        input_array = np.array([input_data])
        input_scaled = scaler.transform(input_array)
        
        # Prediction - CLASS 0 IS DISEASE in this UCI cohort recalibration
        prediction_prob = model.predict_proba(input_scaled)[0][0]
        # prediction = int(model.predict(input_scaled)[0]) 
        # In this reversed mapping, we check if prob of 0 > prob of 1
        prediction = 1 if prediction_prob > 0.5 else 0 
        
        # SHAP Values for Explainability
        shap_values = explainer.shap_values(input_scaled)
        
        # Handle SHAP output format (can be a list for multi-class)
        # For RF binary, shap_values might be [vals_class0, vals_class1]
        if isinstance(shap_values, list):
            current_shap = shap_values[1][0].tolist() # Class 1 (Heart Disease)
        else:
            current_shap = shap_values[0].tolist()

        # Combine features with their SHAP values for visualization
        explanation = []
        for name, val in zip(feature_names, current_shap):
            explanation.append({"feature": name, "impact": val})
            
        # Get Recommendations
        recommendations = get_recommendations(data, prediction_prob)
        
        return jsonify({
            "status": "success",
            "prediction": prediction,
            "probability": round(prediction_prob * 100, 2),
            "explanation": explanation,
            "recommendations": recommendations
        })
        
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

def get_recommendations(data, prob):
    prob_pct = prob * 100
    
    # Stratified Clinical Action & Exercise
    if prob_pct < 35:
        risk_level = "No / Low"
        diagnosis_label = "No / Low Risk - Negative Indicators"
        clinical_action = "Optimal Health: Routine lifestyle maintenance. Screening every 12-24 months."
        exercises = ["150 min/week moderate aerobic", "Yoga for stress management", "Resistance training 2x/week"]
        diet_eat = ["Leafy Greens (Nitrates for BP)", "Berries (Antioxidants)", "Whole Grains (Fiber)", "Legumes (Plant Protein)", "Nuts/Seeds (Omega-3s)"]
        diet_avoid = ["Refined Sugars", "Processed Snacks", "Excessive Caffeine", "High Sodium foods", "Trans-fats"]
    elif prob_pct < 65:
        risk_level = "Moderate"
        diagnosis_label = "Moderate Risk - Progressing Pattern (Reversible)"
        clinical_action = "Precaution: Schedule a Stress Test within 30 days."
        exercises = ["30 min brisk walking 5 days/week", "Controlled breathing"]
        diet_eat = ["Oats/Barley (Beta-glucan)", "Avocados (Mono-fats)", "Soy/Tofu (Isoflavones)", "Garlic/Onions (Allicin)", "Flaxseeds (Alpha-linolenic)"]
        diet_avoid = ["Tropical Oils (Coconut/Palm)", "Vegan Junk Food", "Refined Carbs", "Excess Salt", "Sweetened plant milks"]
    else:
        risk_level = "High"
        diagnosis_label = "High Risk - Potential Irreversible Condition (Urgent)"
        clinical_action = "Urgent: Immediate consultation with a Cardiologist recommended."
        exercises = ["Supervised cardiac rehab only", "Stop if dizziness occurs"]
        diet_eat = ["Beets/Pomegranates (Nitrates)", "Walnuts (Vascular health)", "Steamed Greens (Low-fat)", "Hibiscus Tea (BP Support)", "Low-Glycemic Fruits"]
        diet_avoid = ["Added Salts/Sodium", "All Saturated Fats", "Canned/Processed foods", "Refined Flour", "Coconut Milk/Oil"]

    return {
        "risk_level": risk_level,
        "diagnosis_label": diagnosis_label,
        "clinical_action": clinical_action,
        "diet_to_eat": diet_eat,
        "diet_to_avoid": diet_avoid,
        "supplements": [
            "Vitamin B12 (Essential for vegan diets)",
            "Algae-based Omega-3 (DHA/EPA equivalent)",
            "Vitamin D3 (Vegan-sourced)"
        ],
        "lifestyle_exercises": exercises
    }

from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
import random

# Load Dual Generative Engines
gan_engine = None
vae_engine = None

try:
    if os.path.exists('models/pulse_ai_ctgan.pkl'):
        gan_engine = CTGANSynthesizer.load('models/pulse_ai_ctgan.pkl')
    if os.path.exists('models/pulse_ai_tvae.pkl'):
        vae_engine = TVAESynthesizer.load('models/pulse_ai_tvae.pkl')
    print("✅ Dual Generative Engines (GAN + VAE) Loaded.")
except Exception as e:
    print(f"⚠️ Simulation engines failed to load: {e}")

@app.route('/generate_sample', methods=['POST'])
def generate_sample():
    # 1. Pick a random engine for diversity
    engines = [e for e in [gan_engine, vae_engine] if e is not None]
    if not engines:
        return jsonify({"status": "error", "message": "Simulation engines not loaded"}), 500
    
    current_engine = random.choice(engines)
    risk_type = request.json.get('target', 'low') # 'none', 'low', 'moderate', 'high'
    
    try:
        # 2. AUDITION: Sample a larger cohort to ensure we have candidates in every range
        batch = current_engine.sample(num_rows=500)
        
        # 3. VERIFY: Run the actual model on these candidates
        batch_features = batch[feature_names]
        batch_scaled = scaler.transform(batch_features)
        probs = model.predict_proba(batch_scaled)[:, 0] # TARGET CLASS 0 FOR RISK
        batch['pulse_probability'] = probs
        
        # 4. SELECT: Use strict clinical range filtering with ANCHOR INJECTION
        selected = pd.DataFrame()
        
        # Helper to anchor features for guaranteed range matching
        def anchor_profile(row, risk):
            r = row.copy()
            if risk == 'none':
                r['ca'] = 0; r['oldpeak'] = 0.0; r['cp'] = 0; r['thal'] = 1; r['exang'] = 0
                if r['age'] > 45: r['age'] = 40
            elif risk == 'low':
                r['ca'] = 0; r['oldpeak'] = 0.2; r['exang'] = 0; r['cp'] = 1; r['thal'] = 2
            elif risk == 'high':
                r['ca'] = 2; r['oldpeak'] = 3.0; r['thal'] = 3; r['exang'] = 1; r['cp'] = 3
                if r['age'] < 55: r['age'] = 65
            return r

        if risk_type in ['none', 'low']:
            # Consolidated Healthy Anchor (Target < 35%)
            candidate = batch.sort_values(by='pulse_probability').head(1).iloc[0]
            selected = pd.DataFrame([anchor_profile(candidate, 'none')])
                
        elif risk_type == 'moderate':
            # Target 35-65% (Moderate Risk Range)
            sub_batch = batch[(batch['pulse_probability'] >= 0.35) & (batch['pulse_probability'] < 0.65)]
            selected = sub_batch.sample(1) if not sub_batch.empty else batch.loc[[batch['pulse_probability'].sub(0.60).abs().idxmin()]]
                
        elif risk_type == 'high':
            # Target > 65% (High Risk Range)
            candidate = batch.sort_values(by='pulse_probability', ascending=False).head(1).iloc[0]
            selected = pd.DataFrame([anchor_profile(candidate, 'high')])
        else:
            selected = batch.sample(1)
            
        # Clean up for JSON
        sample_dict = selected.iloc[0].to_dict()
        # Remove helper columns
        sample_dict.pop('pulse_probability', None)
        sample_dict.pop('diff', None)
        
        # Convert types for JSON compatibility
        for k, v in sample_dict.items():
            if isinstance(v, (np.int64, np.float64)):
                sample_dict[k] = float(v)
        
        return jsonify(sample_dict)
    except Exception as e:
        print(f"❌ Simulation Engine Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 400

def draw_clinical_background(canvas, doc):
    canvas.saveState()
    # Subtle light background for a premium feel
    canvas.setFillColor(colors.HexColor("#f8fafc"))
    canvas.rect(0, 0, letter[0], letter[1], fill=1)
    
    # Draw simple Pulse Logo (Heart heartbeat)
    canvas.setStrokeColor(colors.HexColor("#00f2fe"))
    canvas.setLineWidth(2.5)
    
    # Starting coordinates for a centered-ish logo at top left
    x, y = 30, letter[1] - 40
    path = canvas.beginPath()
    path.moveTo(x, y)
    path.lineTo(x+15, y) # flat
    path.lineTo(x+20, y+15) # Up
    path.lineTo(x+25, y-20) # Down deep
    path.lineTo(x+30, y+25) # Up peak
    path.lineTo(x+35, y) # Back to baseline
    path.lineTo(x+55, y) # flat
    canvas.drawPath(path)
    
    # Text next to logo
    canvas.setFont("Helvetica-Bold", 14)
    canvas.setFillColor(colors.HexColor("#1e293b"))
    canvas.drawString(x + 60, y - 5, "Pulse AI | Diagnostic Report")
    
    canvas.restoreState()

@app.route('/generate_report', methods=['POST'])
def generate_report():
    data = request.json
    buffer = BytesIO()
    
    # Tighter margins to ensure 1-page fit
    doc = SimpleDocTemplate(buffer, pagesize=letter, topMargin=60, bottomMargin=40, leftMargin=40, rightMargin=40)
    elements = []
    styles = getSampleStyleSheet()
    
    # Refined Styles
    title_style = ParagraphStyle('Title', parent=styles['Heading1'], fontSize=22, textColor=colors.HexColor("#0f172a"), alignment=0, spaceAfter=20)
    heading_style = ParagraphStyle('Heading', parent=styles['Heading2'], fontSize=14, textColor=colors.HexColor("#4facfe"), spaceBefore=10, spaceAfter=8)
    normal_style = ParagraphStyle('Normal', parent=styles['Normal'], fontSize=9, leading=12)
    
    try:
        recs = data.get('recommendations', {})
        # Primary Title (The "Pulse AI" text is now in the header function)
        elements.append(Paragraph("Clinical Cardiovascular Assessment", title_style))
        elements.append(Spacer(1, 10))
        
        # Patient Profile Table
        elements.append(Paragraph("Patient Clinical Profile", heading_style))
        profile_data = [
            ["Clinical Attribute", "Value", "Notes"],
            ["Patient Age", str(data.get('age', 'N/A')), "Years"],
            ["Biological Sex", "Male" if str(data.get('sex')) == '1' else "Female", "Gender"],
            ["Probability Score", f"{data.get('probability', '0')}%", "Model Prediction"],
            ["Clinical Diagnosis", Paragraph(recs.get('diagnosis_label', 'Clinical Analysis Complete'), normal_style), "AI Analysis"]
        ]
        t = Table(profile_data, colWidths=[160, 200, 160])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#0f172a")),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 10),
        ]))
        elements.append(t)
        
        # Dietary Prescription (Vegan)
        elements.append(Paragraph("Vegan Dietary Prescription", heading_style))
        
        diet_data = [
            [Paragraph("<b>✅ FOODS TO PRIORITIZE</b>", normal_style), Paragraph("<b>❌ FOODS TO AVOID</b>", normal_style)]
        ]
        
        eat_list = recs.get('diet_to_eat', ["Fresh Vegetables", "Leafy Greens", "Whole Grains"])
        avoid_list = recs.get('diet_to_avoid', ["Processed Foods", "Animal Fats", "High Sodium"])
        
        max_len = max(len(eat_list), len(avoid_list))
        for i in range(max_len):
            eat = eat_list[i] if i < len(eat_list) else ""
            avoid = avoid_list[i] if i < len(avoid_list) else ""
            diet_data.append([Paragraph(f"• {eat}", normal_style), Paragraph(f"• {avoid}", normal_style)])

        dt = Table(diet_data, colWidths=[262, 262])
        dt.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('BACKGROUND', (0, 0), (0, 0), colors.HexColor("#f0fdf4")),
            ('BACKGROUND', (1, 0), (1, 0), colors.HexColor("#fef2f2")),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        elements.append(dt)

        # Combined Exercise & Actions to save space
        elements.append(Paragraph("Clinical Recovery & Exercise Plan", heading_style))
        
        action_items = []
        action_items.append(Paragraph(f"<b>Primary Clinical Action:</b> {recs.get('clinical_action', 'Immediate Physician Consultation')}", normal_style))
        action_items.append(Paragraph(f"<b>Cardiac Protocol:</b> {', '.join(recs.get('lifestyle_exercises', ['Brisk Walking', 'Yoga']))}", normal_style))
        action_items.append(Paragraph(f"<b>Supplements:</b> {', '.join(recs.get('supplements', ['Omega-3', 'Vitamin D']) or ['General Wellness Support'])}", normal_style))
        
        for item in action_items:
            elements.append(item)
            elements.append(Spacer(1, 4))

        # FOOTER: Disclaimer (Reduced Spacer to stay on 1 page)
        elements.append(Spacer(1, 15))
        elements.append(Paragraph("<b>Disclaimer:</b> This Pulse AI report is a screening tool. It does not replace professional medical diagnosis.", ParagraphStyle('Disc', fontSize=7, textColor=colors.grey, italics=True)))
        
        # Build with background function
        doc.build(elements, onFirstPage=draw_clinical_background, onLaterPages=draw_clinical_background)
        
        buffer.seek(0)
        return send_file(buffer, as_attachment=True, download_name="PulseAI_Patient_Report.pdf", mimetype='application/pdf')
    except Exception as e:
        print(f"❌ PDF Generation Failure: {e}")
        return jsonify({"status": "error", "message": "Failed to generate report. Details logged."}), 500

@app.route('/results')
def results_page():
    return render_template('results.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        user_message = data.get('message', '')
        patient_context = data.get('patient_context', {})
        user_key = data.get('api_key', '').strip()
        
        # API Key precedence: user-supplied input, then system environment variable
        api_key = user_key or os.environ.get('GEMINI_API_KEY', '')
        
        # Instruct Gemini as an elite medical cardiology assistant
        system_instruction = (
            "You are Pulse AI, a highly specialized clinical cardiology assistant. "
            "You assist physicians with interpreting patient parameters and planning cardiovascular recovery. "
            "Always be clinically precise, structured, and helpful. "
            f"Active Patient Context: {json.dumps(patient_context)}. "
            "Keep answers concise (1-3 paragraphs) and formatted in clear Markdown bullet points where relevant."
        )
        
        prompt = f"{system_instruction}\n\nUser Question: {user_message}"
        
        if api_key:
            # Call Google Gemini API directly via standard library urllib
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
            headers = {'Content-Type': 'application/json'}
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ]
            }
            
            import urllib.request
            req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers=headers, method='POST')
            with urllib.request.urlopen(req, timeout=10) as response:
                res_data = json.loads(response.read().decode('utf-8'))
                reply = res_data['candidates'][0]['content']['parts'][0]['text']
        else:
            # Safe high-fidelity medical mock fallback
            reply = mock_clinical_assistant(user_message, patient_context)
            
        return jsonify({"status": "success", "reply": reply})
        
    except Exception as e:
        return jsonify({"status": "error", "message": f"AI Engine error: {str(e)}"}), 500

def mock_clinical_assistant(query, context):
    q = query.lower()
    age = context.get('age', 54)
    bp = context.get('trestbps', 130)
    chol = context.get('chol', 240)
    
    if "risk" in q or "probability" in q:
        return (
            "### Clinical Cardiology Evaluation\n"
            f"- **Age Context:** Patient age of {age} represents standard structural cardiovascular risk.\n"
            f"- **Vitals Interpretation:** Resting Blood Pressure of {bp} mmHg and Cholesterol of {chol} mg/dL are key markers.\n"
            "- **CDSS Assessment:** We recommend prioritizing cardiac stress tests and monitoring exercise ECG slopes to evaluate coronary blood flow."
        )
    elif "exercise" in q or "sport" in q or "workout" in q:
        return (
            "### Cardiovascular Exercise Recommendations\n"
            "- **Daily Activity:** Target 30-45 minutes of aerobic training (brisk walking, cycling) daily.\n"
            "- **Precautionary:** If exercise angina is suspected, physical exertion should be supervised.\n"
            "- **BP Limit:** Keep heart rate under maximum thresholds if resting blood pressure exceeds 140 mmHg."
        )
    elif "food" in q or "diet" in q or "eat" in q:
        return (
            "### Prescribed Dietary Adjustments\n"
            "- **Increase:** Nitric-oxide dilating ingredients (beets, leafy greens, garlic) to support blood pressure drop.\n"
            "- **Eliminate:** Deep-fried foods, saturated trans-fats, processed meat, and high-sodium counts (>1.5g/day).\n"
            "- **Supplements:** Daily Omega-3 EPA/DHA fatty acids and CoQ10 cellular support."
        )
    elif "reverse" in q or "cure" in q or "heal" in q or "improve" in q:
        return (
            "### Plaque Reversal & Recovery Guidance\n"
            "- **Plaque Stabilization:** While calcified arterial plaques cannot be fully 'cured,' active progression can be halted and soft fibrous plaques can be partially reversed.\n"
            "- **Lipid Optimization:** Lowering LDL cholesterol under **70 mg/dL** (or ideally **55 mg/dL** for high-risk patients) via strict whole-food plant-based nutrition and statins is clinical priority #1.\n"
            "- **Lifestyle Reversal**: Intensive clinical trials (such as Ornish & Esselstyn trials) demonstrate that strict physical activity, high-antioxidant diets, stress management, and tobacco cessation actively repair endothelial function over 12-24 months."
        )
    elif "pain" in q or "angina" in q or "symptom" in q or "breath" in q:
        return (
            "### Clinical Symptom Triage Protocol\n"
            "- **Chest Pain (Angina):** Angina represents myocardial oxygen demand exceeding supply. Stable angina can be managed, but new or worsening symptoms must be triaged immediately.\n"
            "- **Dyspnea (Shortness of Breath):** Can indicate diastolic dysfunction or early heart failure vectors. Monitored resting periods are required.\n"
            "- **Emergency Triggers:** If chest pressure radiates to the left arm, neck, or jaw, or is accompanied by cold sweat, immediately utilize the top red bar to connect with the **Cardiac Emergency Desk (911)**."
        )
    elif "med" in q or "statin" in q or "drug" in q or "pill" in q or "blocker" in q:
        return (
            "### Cardiovascular Pharmacotherapy Overview\n"
            "- **Lipid Management (Statins):** Atorvastatin or Rosuvastatin stabilizes coronary plaques, preventing rupture, and drastically reduces LDL production.\n"
            "- **Pressure Regulation (ACE inhibitors/ARBs):** Lisinopril or Losartan relaxes arterial walls and reduces mechanical strain on the left ventricle.\n"
            "- **Heart Rate Controls (Beta-blockers):** Metoprolol reduces myocardial oxygen consumption by keeping resting heart rate under optimal clinical ranges."
        )
    else:
        return (
            "### Clinical Cardiology Assistant\n"
            f"Active patient context loaded (Age: {age}, Blood Pressure: {bp} mmHg, Cholesterol: {chol} mg/dL).\n"
            "- **Status:** The extracted diagnostic metrics are ready for clinical auditing.\n"
            "- **Guidance:** Ask me questions about risk calculations, Physical Activity boundaries, dietary adjustments, symptoms, or disease reversal pathways."
        )

if __name__ == '__main__':
    app.run(debug=True, port=5000)
