# pollution_app_fixed.py - UPDATED VERSION (No TensorFlow dependency)
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import joblib
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')

# Set page configuration
st.set_page_config(
    page_title="Delhi-NCR Pollution Monitoring",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2e86ab;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
    }
    .prediction-box {
        background-color: #e6f3ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d1edff;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #0d6efd;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class PollutionPredictor:
    def __init__(self):
        self.models_loaded = False
        self.loading_message = st.sidebar.empty()
        
        try:
            # Check if model files exist
            required_files = [
                'random_forest_model.pkl',
                'scaler.pkl', 
                'feature_columns.pkl'
            ]
            
            missing_files = []
            for file in required_files:
                if not os.path.exists(file):
                    missing_files.append(file)
            
            if missing_files:
                self.loading_message.error(f"❌ Missing files: {', '.join(missing_files)}")
                st.sidebar.info("💡 Run 'python fix_models.py' first to create model files")
                return
            
            # Load models
            self.loading_message.info("🔄 Loading models...")
            
            self.rf_model = joblib.load('random_forest_model.pkl')
            self.scaler = joblib.load('scaler.pkl')
            self.feature_columns = joblib.load('feature_columns.pkl')
            
            self.models_loaded = True
            self.loading_message.success("✅ All models loaded successfully!")
            
        except Exception as e:
            self.loading_message.error(f"❌ Error loading models: {str(e)}")
            st.sidebar.info("💡 Try running 'python fix_models.py' to create compatible models")

    def predict_aqi(self, features_dict):
        if not self.models_loaded:
            return None
            
        try:
            # Create feature array in correct order
            feature_array = np.array([[features_dict[col] for col in self.feature_columns]])
            
            # Scale features
            feature_array_scaled = self.scaler.transform(feature_array)
            
            # Predict
            prediction = self.rf_model.predict(feature_array_scaled)[0]
            
            return max(0, prediction)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None
    
    def fallback_prediction(self, features_dict):
        """Fallback prediction when model has compatibility issues"""
    # Simple weighted average based on feature importance
        base_aqi = (
        features_dict.get('stubble_burning', 0) * 2.5 +
        features_dict.get('traffic_intensity', 0) * 1.2 +
        features_dict.get('industrial_activity', 0) * 1.8 +
        features_dict.get('construction_dust', 0) * 1.5 -
        features_dict.get('wind_speed', 0) * 0.8 +
        np.random.normal(0, 5)
        )
        return max(50, base_aqi)

    def get_aqi_category(self, aqi):
        if aqi <= 50:
            return "Good", "🟢", "Excellent air quality", "green", 1
        elif aqi <= 100:
            return "Satisfactory", "🟡", "Acceptable air quality", "yellow", 2
        elif aqi <= 200:
            return "Moderate", "🟠", "Breathing discomfort to sensitive people", "orange", 3
        elif aqi <= 300:
            return "Poor", "🔴", "Breathing discomfort to all", "red", 4
        elif aqi <= 400:
            return "Very Poor", "🟣", "Respiratory illness on prolonged exposure", "purple", 5
        else:
            return "Severe", "⚫", "Health impacts even on healthy people", "black", 6

def main():
    st.markdown('<h1 class="main-header">🌫️ Delhi-NCR Pollution Monitoring & Forecasting</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666;">AI-Driven Pollution Source Identification & Health Advisory System</p>', unsafe_allow_html=True)
    
    # Initialize predictor
    predictor = PollutionPredictor()
    
    # Sidebar for navigation
    st.sidebar.title("🧭 Navigation")
    app_mode = st.sidebar.selectbox("Choose App Mode", 
        ["🏠 Dashboard", "🔮 AQI Prediction", "📊 Source Analysis", "⚖️ Policy Dashboard", "❤️ Health Advisory"])
    
    # Show file status
    st.sidebar.markdown("---")
    st.sidebar.subheader("📁 Model Status")
    
    model_files = {
        'Random Forest Model': 'random_forest_model.pkl',
        'Scaler': 'scaler.pkl',
        'Feature Columns': 'feature_columns.pkl'
    }
    
    for model_name, file_name in model_files.items():
        if os.path.exists(file_name):
            st.sidebar.success(f"✅ {model_name}")
        else:
            st.sidebar.error(f"❌ {model_name}")
    
    if app_mode == "🏠 Dashboard":
        show_dashboard(predictor)
    elif app_mode == "🔮 AQI Prediction":
        show_prediction_interface(predictor)
    elif app_mode == "📊 Source Analysis":
        show_source_analysis(predictor)
    elif app_mode == "⚖️ Policy Dashboard":
        show_policy_dashboard(predictor)
    else:
        show_health_recommendations()

def show_dashboard(predictor):
    st.markdown('<h2 class="sub-header">📊 Pollution Dashboard Overview</h2>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current AQI", "156", "12% ↗", delta_color="inverse")
    with col2:
        st.metric("Primary Source", "Stubble", "35% contribution")
    with col3:
        st.metric("Wind Speed", "8 km/h", "Good dispersion")
    with col4:
        st.metric("Health Impact", "Moderate", "Sensitive groups affected")
    
    st.markdown("---")
    
    # Quick prediction section
    st.subheader("🚀 Quick AQI Prediction")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stubble = st.slider("Stubble Burning Intensity", 0, 20, 8, key="dashboard_stubble")
        traffic = st.slider("Traffic Intensity", 0, 100, 65, key="dashboard_traffic")
        wind = st.slider("Wind Speed (km/h)", 0, 30, 12, key="dashboard_wind")
    
    with col2:
        industrial = st.slider("Industrial Activity", 0, 100, 45, key="dashboard_industrial")
        construction = st.slider("Construction Dust", 0, 100, 30, key="dashboard_construction")
        temp = st.slider("Temperature (°C)", 0, 40, 28, key="dashboard_temp")
    
    if st.button("🔍 Quick Predict", type="primary"):
        if predictor.models_loaded:
            features = {
                'temperature': temp,
                'humidity': 65,  # Default value
                'wind_speed': wind,
                'stubble_burning': stubble,
                'traffic': traffic,
                'industrial': industrial,
                'construction': construction,
                'month': datetime.now().month
            }
            
            prediction = predictor.predict_aqi(features)
            if prediction is not None:
                category, emoji, advice, color, level = predictor.get_aqi_category(prediction)
                
                st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
                st.metric("Predicted AQI", f"{prediction:.1f}")
                st.write(f"**Air Quality:** {emoji} {category}")
                st.write(f"**Advisory:** {advice}")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Models not loaded. Please check the model status in sidebar.")

def show_prediction_interface(predictor):
    st.markdown('<h2 class="sub-header">🔮 Real-time AQI Prediction</h2>', unsafe_allow_html=True)
    
    if not predictor.models_loaded:
        st.error("""
        **Models not loaded!** Please ensure you have the following files in the same directory:
        - `random_forest_model.pkl`
        - `scaler.pkl`
        - `feature_columns.pkl`
        
        💡 Run 'python fix_models.py' first to create these files.
        """)
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌤️ Environmental Conditions")
        
        temperature = st.slider("Temperature (°C)", 0.0, 45.0, 28.0)
        humidity = st.slider("Humidity (%)", 0.0, 100.0, 65.0)
        wind_speed = st.slider("Wind Speed (km/h)", 0.0, 50.0, 12.0)
        rainfall = st.slider("Rainfall (mm)", 0.0, 50.0, 0.0)
    
    with col2:
        st.subheader("🏭 Pollution Sources")
        
        stubble_burning = st.slider("Stubble Burning Intensity", 0, 20, 8)
        traffic_intensity = st.slider("Traffic Intensity", 0, 100, 75)
        industrial_activity = st.slider("Industrial Activity", 0, 100, 55)
        construction_dust = st.slider("Construction Dust", 0, 100, 35)
        
        current_date = st.date_input("Date", datetime.now())
        month = current_date.month
    
    # Prepare features
    features = {
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'stubble_burning': stubble_burning,
        'traffic': traffic_intensity,
        'industrial': industrial_activity,
        'construction': construction_dust,
        'month': month
    }
    
    # Prediction button
    if st.button("🎯 Predict AQI", type="primary", use_container_width=True):
        with st.spinner("Analyzing environmental conditions..."):
            prediction = predictor.predict_aqi(features)
            
            if prediction is not None:
                category, emoji, advice, color, level = predictor.get_aqi_category(prediction)
                
                # Display results
                st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted AQI", f"{prediction:.1f}")
                    st.write(f"**Air Quality Category:** {emoji} {category}")
                    st.write(f"**Health Impact:** {advice}")
                
                with col2:
                    # Source contribution estimation
                    st.write("**Estimated Source Contributions:**")
                    total = stubble_burning * 2.5 + traffic_intensity * 1.2 + industrial_activity * 1.8 + construction_dust * 1.5
                    if total > 0:
                        st.write(f"• Stubble Burning: {(stubble_burning * 2.5 / total * 100):.1f}%")
                        st.write(f"• Traffic: {(traffic_intensity * 1.2 / total * 100):.1f}%")
                        st.write(f"• Industrial: {(industrial_activity * 1.8 / total * 100):.1f}%")
                        st.write(f"• Construction: {(construction_dust * 1.5 / total * 100):.1f}%")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Show immediate recommendations
                show_immediate_recommendations(prediction, category, level)
                
                # Show AQI scale
                show_aqi_scale(prediction)

def show_immediate_recommendations(aqi, category, level):
    st.subheader("🚨 Immediate Health Advisory")
    
    if level == 1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write("✅ **Excellent Air Quality**")
        st.write("- Perfect for all outdoor activities")
        st.write("- Ideal for exercise and sports")
        st.write("- No restrictions needed")
        st.markdown('</div>', unsafe_allow_html=True)
    elif level == 2:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write("⚠️ **Satisfactory Air Quality**")
        st.write("- Generally acceptable for most activities")
        st.write("- Minor discomfort for sensitive individuals")
        st.write("- Consider reducing prolonged exertion if sensitive")
        st.markdown('</div>', unsafe_allow_html=True)
    elif level == 3:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write("🔶 **Moderate Air Quality**")
        st.write("- Children, elderly, and people with respiratory issues should limit outdoor activities")
        st.write("- Consider wearing masks if sensitive to air pollution")
        st.write("- Reduce prolonged outdoor exertion")
        st.markdown('</div>', unsafe_allow_html=True)
    elif level == 4:
        st.markdown('<div class="danger-box">', unsafe_allow_html=True)
        st.write("🔴 **Poor Air Quality**")
        st.write("- Avoid outdoor activities")
        st.write("- Wear N95 masks if going outside")
        st.write("- Use air purifiers indoors")
        st.write("- Sensitive groups should stay indoors")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="danger-box">', unsafe_allow_html=True)
        st.write("🚨 **Very Poor to Severe Air Quality**")
        st.write("- Stay indoors as much as possible")
        st.write("- Use high-efficiency air purifiers")
        st.write("- Avoid all physical exertion")
        st.write("- Consider relocating if conditions persist")
        st.markdown('</div>', unsafe_allow_html=True)

def show_aqi_scale(current_aqi):
    st.subheader("📊 AQI Scale Reference")
    
    aqi_ranges = [
        (0, 50, "Good", "🟢", "#00E400"),
        (51, 100, "Satisfactory", "🟡", "#FFFF00"),
        (101, 200, "Moderate", "🟠", "#FF7E00"),
        (201, 300, "Poor", "🔴", "#FF0000"),
        (301, 400, "Very Poor", "🟣", "#8F3F97"),
        (401, 500, "Severe", "⚫", "#7E0023")
    ]
    
    # Create visual AQI scale
    fig, ax = plt.subplots(figsize=(12, 3))
    
    for i, (low, high, label, emoji, color) in enumerate(aqi_ranges):
        ax.barh(0, high-low, left=low, color=color, alpha=0.8, edgecolor='black')
        ax.text((low + high) / 2, 0, f"{emoji}\n{label}", 
                ha='center', va='center', fontweight='bold', fontsize=9)
    
    # Mark current AQI
    ax.axvline(x=current_aqi, color='white', linestyle='--', linewidth=3)
    ax.text(current_aqi, 0.5, f'Predicted: {current_aqi:.0f}', 
            rotation=90, va='bottom', fontweight='bold', color='white',
            bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))
    
    ax.set_xlim(0, 500)
    ax.set_ylim(-1, 1)
    ax.set_yticks([])
    ax.set_xlabel('AQI Value')
    ax.set_title('Air Quality Index (AQI) Scale', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)

def show_source_analysis(predictor):
    st.markdown('<h2 class="sub-header">📊 Pollution Source Analysis</h2>', unsafe_allow_html=True)
    
    st.info("Adjust the sliders to analyze how different pollution sources contribute to overall AQI.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Source Intensities")
        stubble_burning = st.slider("Stubble Burning", 0, 20, 8, key="source_stubble")
        traffic_emissions = st.slider("Traffic Emissions", 0, 100, 65, key="source_traffic")
        industrial_activity = st.slider("Industrial Activity", 0, 100, 45, key="source_industrial")
        construction_dust = st.slider("Construction Dust", 0, 100, 30, key="source_construction")
        
        st.subheader("🌍 Environmental Factors")
        wind_speed = st.slider("Wind Speed", 0, 30, 12, key="source_wind")
        rainfall = st.slider("Rainfall", 0, 50, 0, key="source_rain")
        temperature = st.slider("Temperature", 0, 40, 28, key="source_temp")
    
    with col2:
        # Calculate source contributions
        source_weights = {
            'Stubble Burning': 2.5,
            'Traffic Emissions': 1.2,
            'Industrial Activity': 1.8,
            'Construction Dust': 1.5
        }
        
        # Map slider values to source intensities
        source_intensities = {
            'Stubble Burning': stubble_burning,
            'Traffic Emissions': traffic_emissions,
            'Industrial Activity': industrial_activity,
            'Construction Dust': construction_dust
        }
        
        contributions = {}
        total_pollution = 0
        
        for source, weight in source_weights.items():
            intensity = source_intensities[source]
            contribution = intensity * weight
            contributions[source] = contribution
            total_pollution += contribution
        
        # Adjust for environmental factors
        dispersion = wind_speed * 0.8 + rainfall * 2
        adjusted_pollution = max(50, total_pollution - dispersion + np.random.normal(0, 10))
        
        # Calculate percentages
        if total_pollution > 0:
            source_percentages = {k: (v / total_pollution) * 100 for k, v in contributions.items()}
        else:
            source_percentages = {k: 0 for k in contributions.keys()}
        
        st.metric("Estimated AQI", f"{adjusted_pollution:.1f}")
        st.metric("Dispersion Factor", f"{(dispersion):.1f}")
        
        # Show individual contributions
        st.subheader("📋 Source Breakdown")
        for source, percentage in source_percentages.items():
            st.write(f"• {source}: {percentage:.1f}%")
    
    # Display charts
    st.subheader("📈 Source Contribution Analysis")
    
    col3, col4 = st.columns(2)
    
    with col3:
        # Pie chart
        fig1, ax1 = plt.subplots(figsize=(8, 8))
        colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4']
        ax1.pie(source_percentages.values(), labels=source_percentages.keys(), 
                autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title('Pollution Source Contribution')
        st.pyplot(fig1)
    
    with col4:
        # Bar chart
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sources = list(source_percentages.keys())
        percentages = list(source_percentages.values())
        
        bars = ax2.bar(sources, percentages, color=colors)
        ax2.set_ylabel('Contribution (%)')
        ax2.set_title('Pollution Source Breakdown')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, percentages):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        st.pyplot(fig2)
    
    # Policy recommendations
    st.subheader("🎯 Targeted Intervention Recommendations")
    
    max_source = max(source_percentages, key=source_percentages.get)
    max_percentage = source_percentages[max_source]
    
    if max_percentage > 35:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write(f"🚨 **{max_source} is the dominant contributor ({max_percentage:.1f}%)**")
        
        if max_source == 'Stubble Burning':
            st.write("- Implement stubble management subsidies")
            st.write("- Promote alternative uses for crop residue")
            st.write("- Enforce burning bans during critical periods")
            st.write("- Provide farmers with stubble management equipment")
        elif max_source == 'Traffic Emissions':
            st.write("- Implement odd-even vehicle policy")
            st.write("- Promote public transportation and carpooling")
            st.write("- Encourage electric vehicle adoption")
            st.write("- Improve traffic management systems")
        elif max_source == 'Industrial Activity':
            st.write("- Enforce stricter emission standards")
            st.write("- Promote cleaner production technologies")
            st.write("- Implement real-time emission monitoring")
            st.write("- Encourage shift to cleaner fuels")
        else:
            st.write("- Enforce dust control measures at construction sites")
            st.write("- Mandate use of dust suppression systems")
            st.write("- Implement construction activity restrictions during high pollution days")
            st.write("- Promote use of covered transportation for construction materials")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    if wind_speed < 10 and rainfall == 0:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write("💨 **Poor Dispersion Conditions Detected**")
        st.write("- High pollution accumulation expected")
        st.write("- Consider temporary industrial restrictions")
        st.write("- Alert public about poor air quality")
        st.write("- Implement emergency response measures")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Show environmental impact
    st.subheader("🌤️ Environmental Impact Assessment")
    
    env_col1, env_col2, env_col3 = st.columns(3)
    
    with env_col1:
        if wind_speed > 15:
            st.success("💨 **Good Wind Dispersion**")
            st.write("Favorable for pollution dispersion")
        elif wind_speed > 8:
            st.info("💨 **Moderate Wind Dispersion**")
            st.write("Average dispersion conditions")
        else:
            st.warning("💨 **Poor Wind Dispersion**")
            st.write("Pollution may accumulate")
    
    with env_col2:
        if rainfall > 10:
            st.success("🌧️ **Good Rain Scavenging**")
            st.write("Rain helps clear pollutants")
        elif rainfall > 5:
            st.info("🌧️ **Moderate Rain Scavenging**")
            st.write("Some pollution clearing")
        else:
            st.warning("🌧️ **No Rain Scavenging**")
            st.write("No natural cleaning")
    
    with env_col3:
        if temperature > 30:
            st.info("🌡️ **High Temperature**")
            st.write("May increase ozone formation")
        elif temperature > 20:
            st.success("🌡️ **Moderate Temperature**")
            st.write("Stable atmospheric conditions")
        else:
            st.warning("🌡️ **Low Temperature**")
            st.write("May lead to inversion layers")

def show_policy_dashboard(predictor):
    st.markdown('<h2 class="sub-header">⚖️ Policy Effectiveness Dashboard</h2>', unsafe_allow_html=True)
    
    st.info("Monitor the effectiveness of various pollution control policies and their impacts.")
    
    # Policy effectiveness data
    policies = {
        'Odd-Even Vehicle Policy': {
            'effectiveness': 15, 
            'cost': 'Medium', 
            'public_acceptance': 65, 
            'duration': 'Short-term',
            'impact': 'Immediate but temporary reduction in traffic emissions'
        },
        'Stubble Burning Ban': {
            'effectiveness': 25, 
            'cost': 'Low', 
            'public_acceptance': 70, 
            'duration': 'Seasonal',
            'impact': 'Significant reduction during harvest seasons'
        },
        'Industrial Restrictions': {
            'effectiveness': 20, 
            'cost': 'High', 
            'public_acceptance': 60, 
            'duration': 'Long-term',
            'impact': 'Sustainable reduction but economic implications'
        },
        'Construction Regulations': {
            'effectiveness': 10, 
            'cost': 'Medium', 
            'public_acceptance': 75, 
            'duration': 'Long-term',
            'impact': 'Localized improvement around construction sites'
        },
        'Public Transport Incentives': {
            'effectiveness': 12, 
            'cost': 'High', 
            'public_acceptance': 80, 
            'duration': 'Long-term',
            'impact': 'Gradual shift from private vehicles'
        }
    }
    
    # Display policy metrics
    st.subheader("📋 Policy Performance Metrics")
    
    cols = st.columns(4)
    with cols[0]:
        st.write("**📈 Effectiveness (%)**")
        for policy, data in policies.items():
            st.metric(policy, f"{data['effectiveness']}%")
    
    with cols[1]:
        st.write("**💰 Implementation Cost**")
        for policy, data in policies.items():
            color = "🟢" if data['cost'] == 'Low' else "🟡" if data['cost'] == 'Medium' else "🔴"
            st.write(f"**{policy}:** {color} {data['cost']}")
    
    with cols[2]:
        st.write("**👥 Public Acceptance (%)**")
        for policy, data in policies.items():
            st.metric(policy, f"{data['public_acceptance']}%")
    
    with cols[3]:
        st.write("**⏱️ Implementation Duration**")
        for policy, data in policies.items():
            st.write(f"**{policy}:** {data['duration']}")
    
    # AI Policy recommendations
    st.subheader("🤖 AI-Generated Policy Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.write("**🎯 High Priority Actions**")
        st.write("1. **Stubble Management Programs** (Best effectiveness-to-cost ratio)")
        st.write("2. **Public Transport Enhancement** (High public acceptance)")
        st.write("3. **Industrial Emission Monitoring** (Long-term impact)")
        st.write("4. **Emergency Response Plan** (For severe pollution episodes)")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="warning-box">', unsafe_allow_html=True)
        st.write("**📊 Expected Impact Timeline**")
        st.write("- **Immediate (1-4 weeks):** 5-10% AQI reduction")
        st.write("- **Short-term (1-3 months):** 10-15% AQI improvement")
        st.write("- **Medium-term (6 months):** 20-25% AQI improvement") 
        st.write("- **Long-term (1 year+):** 30-40% sustainable improvement")
        st.markdown('</div>', unsafe_allow_html=True)

def show_health_recommendations():
    st.markdown('<h2 class="sub-header">❤️ Health Advisory & Safe Routes</h2>', unsafe_allow_html=True)
    
    st.subheader("👤 Personalized Health Assessment")
    
    # User input for health conditions
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Demographic Information**")
        age = st.selectbox("Age Group", 
                          ["Under 18", "18-30", "31-45", "46-60", "Over 60"])
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        weight = st.slider("Weight (kg)", 30, 120, 70)
        height = st.slider("Height (cm)", 120, 200, 170)
    
    with col2:
        st.write("**Health Conditions**")
        has_asthma = st.checkbox("Asthma or Respiratory Conditions")
        has_heart = st.checkbox("Heart Conditions")
        has_allergies = st.checkbox("Allergies")
        is_pregnant = st.checkbox("Pregnant")
        smoker = st.checkbox("Smoker")
    
    st.write("**Lifestyle Factors**")
    col3, col4 = st.columns(2)
    
    with col3:
        activity_level = st.selectbox("Daily Outdoor Activity Level",
                                    ["Mostly Indoors", "Light (1-2 hours outdoors)",
                                     "Moderate (3-4 hours outdoors)", "High (5+ hours outdoors)"])
        location = st.selectbox("Primary Location",
                               ["Residential Area", "Commercial Area", "Industrial Area", "Mixed Use", "Rural"])
    
    with col4:
        commute_type = st.selectbox("Primary Commute Method",
                                  ["Personal Vehicle", "Public Transport", "Walking", "Cycling", "Mixed"])
        work_hours = st.slider("Daily Outdoor Work Hours", 0, 12, 2)
    
    # Calculate health risk score
    risk_score = 0
    if has_asthma or has_heart: risk_score += 3
    if is_pregnant: risk_score += 2
    if age in ["Under 18", "Over 60"]: risk_score += 2
    if smoker: risk_score += 2
    if has_allergies: risk_score += 1
    if activity_level in ["Moderate", "High"]: risk_score += 1
    if work_hours > 4: risk_score += 1
    if location in ["Industrial Area", "Commercial Area"]: risk_score += 1
    if commute_type in ["Walking", "Cycling"]: risk_score += 1
    
    # Generate recommendations based on risk score
    st.subheader("🎯 Personalized Health Recommendations")
    
    if risk_score >= 8:
        risk_level = "🚨 High Risk"
        color = "danger-box"
    elif risk_score >= 5:
        risk_level = "⚠️ Moderate Risk" 
        color = "warning-box"
    else:
        risk_level = "✅ Low Risk"
        color = "success-box"
    
    st.markdown(f'<div class="{color}">', unsafe_allow_html=True)
    st.write(f"**Risk Assessment:** {risk_level} (Score: {risk_score}/12)")
    
    recommendations = []
    
    if has_asthma or has_heart:
        recommendations.extend([
            "💊 Always carry emergency medications",
            "🏠 Use HEPA air purifiers at home and work",
            "📞 Keep emergency contacts readily accessible",
            "🚑 Have an emergency action plan"
        ])
    
    if is_pregnant:
        recommendations.extend([
            "🤰 Limit exposure to high pollution areas",
            "🕒 Avoid outdoor activities during peak pollution hours (7-10 AM, 5-8 PM)",
            "🌿 Spend time in green spaces with better air quality",
            "🏥 Regular health check-ups"
        ])
    
    if age in ["Under 18", "Over 60"]:
        recommendations.extend([
            "👶👴 Extra caution during high pollution days",
            "🏫 Schools should limit outdoor activities when AQI > 150",
            "🌅 Morning and evening are better for outdoor activities",
            "💪 Maintain good overall health with balanced diet"
        ])
    
    if smoker:
        recommendations.extend([
            "🚭 Consider smoking cessation programs",
            "🏠 Avoid smoking indoors",
            "💨 Smoking increases vulnerability to air pollution effects"
        ])
    
    # General recommendations
    recommendations.extend([
        "😷 Wear N95 masks in high pollution areas (AQI > 150)",
        "🌳 Choose green routes for walking/commuting",
        "💧 Stay hydrated to help body flush toxins", 
        "🏠 Keep windows closed during high pollution days",
        "📱 Monitor real-time AQI updates regularly",
        "🌬️ Use air purifiers in bedrooms and living areas",
        "🚗 Keep car windows closed and use air recirculation",
        "🥗 Maintain a healthy diet rich in antioxidants",
        "💤 Ensure adequate sleep for better immunity"
    ])
    
    for rec in recommendations:
        st.write(f"• {rec}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Safe route suggestions
    st.subheader("🗺️ Smart Route Suggestions")
    
    routes_data = {
        "Morning Commute (7-9 AM)": {
            "safest": "🌿 Green Route via Parks - AQI: 95-120",
            "alternative": "🚇 Metro + Short Walk - AQI: 85-110",
            "avoid": "🛣️ Highway Route - AQI: 180-220",
            "time": "+5-10 minutes",
            "benefit": "60% less pollution exposure"
        },
        "Evening Commute (5-7 PM)": {
            "safest": "🏘️ Residential Areas - AQI: 110-140", 
            "alternative": "🛣️ Expressway (Less Traffic) - AQI: 130-160",
            "avoid": "🏬 Commercial Centers - AQI: 200-250",
            "time": "+8-12 minutes", 
            "benefit": "45% less pollution exposure"
        },
        "Exercise & Walking": {
            "safest": "🌅 Early Morning (6-7 AM) - AQI: 70-90",
            "alternative": "🌃 Late Evening (8-9 PM) - AQI: 90-110",
            "avoid": "☀️ Afternoon (2-4 PM) - AQI: 150-180",
            "time": "Optimal timing",
            "benefit": "50% better air quality"
        },
        "Weekend Activities": {
            "safest": "🌳 Botanical Gardens - AQI: 80-100",
            "alternative": "🛍️ Indoor Malls - AQI: 90-110", 
            "avoid": "🎪 Outdoor Markets - AQI: 200-240",
            "time": "Similar duration",
            "benefit": "70% less pollution exposure"
        }
    }
    
    for activity, info in routes_data.items():
        with st.expander(f"{activity}"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success("✅ **Safest Option**")
                st.write(info['safest'])
                st.caption(f"Time: {info['time']}")
                st.caption(f"Benefit: {info['benefit']}")
            
            with col2:
                st.info("🔄 **Good Alternative**")
                st.write(info['alternative'])
                st.caption("Balanced option")
            
            with col3:
                st.error("❌ **Avoid This Route**")
                st.write(info['avoid'])
                st.caption("High pollution exposure")

if __name__ == "__main__":
    main()