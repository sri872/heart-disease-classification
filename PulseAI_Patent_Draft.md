# PATENT DRAFT: Pulse AI Clinical Decision Support System

## 1. TITLE OF THE INVENTION
**Pulse AI: An Explainable AI-Driven Clinical Decision Support System for Automated Cardiac Risk Stratification and Explainable Diagnostic Reporting.**

---

## 2. PROBLEM STATEMENT / BACKGROUND
Heart disease remains the leading cause of global mortality, responsible for nearly 18 million deaths annually. Early and accurate detection is critical for improving patient outcomes; however, frontline clinicians often face high cognitive loads and time constraints, which can lead to the omission of subtle diagnostic patterns within routine clinical vitals. 

While automated diagnostic tools exist, they frequently operate as "black boxes," providing predictions without clinical justification, which limits provider trust and clinical adoption. Furthermore, the lack of private, non-cloud-based diagnostic tools and the absence of high-fidelity simulation for clinical training are significant gaps in the current medical technology landscape. There is an urgent need for an explainable, privacy-first Clinical Decision Support System (CDSS) that not only predicts risk but provides actionable, transparent clinical reasoning.

---

## 3. THE INVENTION — TECHNICAL AND CLINICAL FACTORS
This invention represents a substantial advancement in cardiac diagnostics by integrating predictive accuracy with clinical interpretability and privacy-preserving generative simulation.

### A. Core Technical Variables
*   **Clinical Input Parameters (13 Features):** Age, Sex, Chest Pain Type (cp), Resting Blood Pressure (trestbps), Serum Cholesterol (chol), Fasting Blood Sugar (fbs), Resting ECG Results (restecg), Maximum Heart Rate (thalach), Exercise Induced Angina (exang), ST Depression (oldpeak), Thalassemia Type (thal), and Number of Major Vessels (ca).
*   **Machine Learning Engine:** 
    *   **Inference Model:** Optimized Random Forest Classifier (n_estimators=100) trained on standardized clinical datasets.
    *   **Interpretability Engine:** Integration of SHAP (SHapley Additive exPlanations) using `TreeExplainer` for per-prediction feature attribution, ensuring every risk score is clinically auditable.
*   **Generative AI Engines (Simulation):** Dual-engine architecture utilizing **CTGAN** (Conditional Tabular GAN) and **TVAE** (Tabular Variational Autoencoder) for high-fidelity, privacy-safe synthetic patient generation.

### B. UI and Workflow Variables
*   **Clinician Dashboard:** A dark-themed, glassmorphic single-page interface designed for low-fatigue high-stakes medical environments.
*   **XAI Visualization:** Real-time horizontal bar charts mapping Shapley values to clinical features to explain risk drivers.
*   **Automated Clinical Reporting:** Server-side PDF generation engine (ReportLab) that transforms AI predictions, XAI visualizations, and tailored clinical protocols into standardized medical documentation.

### C. Clinical Stratification Logic
*   **Three-Tier Risk Index:** Automated mapping of probability scores into actionable clinical buckets:
    *   **No/Low Risk (0-35%):** Routine screening and lifestyle maintenance.
    *   **Moderate Risk (35-65%):** Non-invasive diagnostic follow-ups and secondary prevention.
    *   **High Risk (65-100%):** Immediate cardiologist referral and urgent clinical intervention.

---

## 4. WORKING PROTOTYPE / DESIGN
The existing functional prototype is a locally-hosted Flask web application architecture. It features:
1.  **Secure Authentication:** Physician-centric access control.
2.  **Input Normalization:** `StandardScaler` pipeline for clinical feature alignment.
3.  **XAI Pipeline:** Sub-second generation of feature impact charts.
4.  **Local Ledger:** Browser-side `localStorage` for privacy-preserving patient history (no cloud data exposure).
5.  **Simulation Dashboard:** A dedicated environment for training clinicians using GAN-generated synthetic profiles.

---

## 5. EXISTING DATA
The system is trained on the UCI Heart Disease Dataset, augmented with 2,000 synthetic clinical records generated via the integrated CTGAN/TVAE engines to improve model robustness and edge-case handling.

---

## 6. USE AND DISCLOSURE
*   **Has the invention been shown in conferences?** NO
*   **Attempts to commercialize?** NO
*   **Described in printed publications?** NO
*   **Collaborations?** Internal University Prototype

---

## 7. POTENTIAL CHANCES OF COMMERCIALIZATION
*   **Global Burden:** With over 500 million people living with cardiovascular diseases, the market for rapid screening tools is immense.
*   **Explainability Trend:** Healthcare is shifting away from "black-box" AI towards Explainable AI (XAI), which is a core competitive advantage of Pulse AI.
*   **Privacy-by-Design:** The local-first, no-cloud architecture appeals to hospital systems with strict HIPAA/GDPR data sovereignty requirements.
*   **Adoption Potential:** Capable of being integrated into existing Electronic Health Record (EHR) systems via HL7 FHIR standards.

---

## 8. NOVELTY & FTO POSITION (Prior Art Comparison)

| Sr. No. | Patent ID / Assignee | Abstract / Key Idea | Research Gap / Limitation | Overlap with Invention | Novelty & FTO Position |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | US 20210082565A1 (IBM) | AI-based cardiac risk prediction using EHR data. | Complex enterprise scale; lacks real-time clinician-facing XAI charts. | Cardiac risk prediction. | **Safe** — Pulse AI focused on local-first CDSS with per-prediction SHAP charts. |
| 2 | US 10453573B2 (Google Health) | Neural networks for predicting cardiovascular risk from eye scans. | Requires expensive retinal imaging equipment. | Cardiovascular risk prediction. | **Safe** — Pulse AI uses low-cost routine clinical vitals. |
| 3 | WO 2020210544A1 (Zivitas) | Dashboard for displaying ML-based health risks. | Generic risk display; no integrated GAN-simulation for training. | Health monitoring UI. | **Safe** — Pulse AI uniqueness = Fusion of XAI + Dual GAN simulation + MD reporting. |
| 4 | US 20190287661A1 (HeartFlow) | Modeling blood flow from CT scans using AI. | Highly specific to CT imaging and hemodynamics. | Heart disease AI. | **Safe** — Pulse AI is a general clinical screening tool for intake vitals. |

---

## 9. KEYWORDS
Heart Disease Classification, Clinical Decision Support System (CDSS), Explainable AI (XAI), SHAP Interpretability, Random Forest Classifier, Cardiovascular Risk Stratification, Medical Diagnostic Tool, Privacy-Preserving AI, GAN-based Clinical Simulation, CTGAN, TVAE, Healthcare Informatics, Clinical Reporting Automation, Physician Dashboard, Digital Biomarkers.

---

## 10. FILING OPTIONS
**Filing Level:** Complete Patent Filing (due to end-to-end integration of XAI and Synthetic Data engines in a clinical workflow).

---

## 11. NO OBJECTION CERTIFICATE
*(To be signed by university/external stakeholders as per template)*

---
**Prepared By:** Antigravity AI (Pulse AI Development Team)
**Date:** April 18, 2026
