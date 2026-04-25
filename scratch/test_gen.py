import sdv
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer
import os
import pandas as pd

def test_sampling():
    try:
        if os.path.exists('models/pulse_ai_ctgan.pkl'):
            gan_engine = CTGANSynthesizer.load('models/pulse_ai_ctgan.pkl')
            print("✅ CTGAN Loaded")
            conditions = pd.DataFrame({'target': [0]})
            sample = gan_engine.sample_from_conditions(conditions)
            print("✅ CTGAN Sampled:", sample.shape)
        
        if os.path.exists('models/pulse_ai_tvae.pkl'):
            vae_engine = TVAESynthesizer.load('models/pulse_ai_tvae.pkl')
            print("✅ TVAE Loaded")
            conditions = pd.DataFrame({'target': [1]})
            sample = vae_engine.sample_from_conditions(conditions)
            print("✅ TVAE Sampled:", sample.shape)
            
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_sampling()
