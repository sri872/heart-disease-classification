import os
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

if __name__ == '__main__':
    app.run(debug=True, port=5000)
