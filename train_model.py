import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os

def train_pulse_ai():
    print("🚀 Starting Pulse AI Model Training...")
    
    # URL to the UCI heart disease dataset
    data_url = "https://raw.githubusercontent.com/sharmaroshan/Heart-UCI-Dataset/master/heart.csv"
    
    try:
        real_df = pd.read_csv(data_url)
        print(f"✅ Real Dataset loaded: {len(real_df)} records.")
        
        # Check for synthetic data augmentation
        synthetic_path = 'data/synthetic_heart_data.csv'
        if os.path.exists(synthetic_path):
            synth_df = pd.read_csv(synthetic_path)
            print(f"🧪 Synthetic Dataset loaded: {len(synth_df)} records.")
            df = pd.concat([real_df, synth_df], axis=0).reset_index(drop=True)
            print(f"📊 Merged Dataset: {len(df)} total records for training.")
        else:
            df = real_df
            print("⚠️ Synthetic data not found. Using real data only.")
            
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        # Fallback to simple synthetic data if network fails (unlikely in this env but good practice)
        return

    # Features and Target
    X_real = real_df.drop('target', axis=1)
    y_real = real_df['target']

    # 1. Split REAL data into Train/Test (ALWAYS test on real data)
    X_train_real, X_test_real, y_train_real, y_test_real = train_test_split(X_real, y_real, test_size=0.2, random_state=42)

    # 2. Skip noisy synthetic data for the final clinical core
    X_train, y_train = X_train_real, y_train_real
    print(f"📊 Training on HIGH-PURITY Real Dataset: {len(X_train)} records.")

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test_real) # Evaluate on REAL test set

    # Train Random Forest (Restored Depth)
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Evaluate on real data
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test_real, y_pred)
    print(f"📈 Real-World Performance (on Real Test Set): {acc:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test_real, y_pred))

    # Save artifacts
    joblib.dump(model, 'pulse_ai_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    # Save the feature names for reference in UI
    joblib.dump(X_real.columns.tolist(), 'feature_names.pkl')
    
    print("✅ Model, Scaler, and Metadata saved as 'pulse_ai_model.pkl', 'scaler.pkl', and 'feature_names.pkl'")

if __name__ == "__main__":
    train_pulse_ai()
