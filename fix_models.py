# retrain_streamlit.py
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("🔄 Retraining models for Streamlit compatibility...")

def create_compatible_dataset():
    np.random.seed(42)
    n_samples = 2000
    
    data = {
        'temperature': np.random.normal(25, 8, n_samples),
        'humidity': np.random.normal(60, 20, n_samples),
        'wind_speed': np.random.gamma(2, 2, n_samples),
        'stubble_burning': np.random.poisson(5, n_samples),
        'traffic': np.random.normal(70, 20, n_samples),
        'industrial': np.random.normal(50, 15, n_samples),
        'construction': np.random.normal(30, 10, n_samples),
        'month': np.random.randint(1, 13, n_samples)
    }
    
    # Create realistic AQI
    data['aqi'] = (
        data['stubble_burning'] * 2.5 +
        data['traffic'] * 1.2 + 
        data['industrial'] * 1.8 +
        data['construction'] * 1.5 -
        data['wind_speed'] * 0.8 +
        np.random.normal(0, 15, n_samples)
    )
    
    data['aqi'] = np.maximum(50, np.minimum(500, data['aqi']))
    return pd.DataFrame(data)

# Create dataset
df = create_compatible_dataset()
print(f"✅ Dataset created: {df.shape}")

# Prepare features
feature_columns = ['temperature', 'humidity', 'wind_speed', 'stubble_burning', 
                   'traffic', 'industrial', 'construction', 'month']

X = df[feature_columns]
y = df['aqi']

# Create COMPATIBLE Random Forest (simple parameters)
print("🌲 Training Streamlit-compatible Random Forest...")
rf_model = RandomForestRegressor(
    n_estimators=100,      # Reduced for compatibility
    max_depth=15,          # Reduced for compatibility
    random_state=42,
    n_jobs=-1
    # NO advanced parameters that might cause compatibility issues
)
rf_model.fit(X, y)

# Create scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save models
joblib.dump(rf_model, 'random_forest_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(feature_columns, 'feature_columns.pkl')

print("✅ Streamlit-compatible models saved!")

# Test prediction
test_features = {
    'temperature': 25, 'humidity': 60, 'wind_speed': 5,
    'stubble_burning': 8, 'traffic': 75,
    'industrial': 55, 'construction': 35, 'month': 11
}

feature_array = np.array([[test_features[col] for col in feature_columns]])
prediction = rf_model.predict(feature_array)[0]
print(f"🧪 Test prediction: {prediction:.2f} AQI")

# Show feature importance
importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n🔍 Feature Importance:")
print(importance)