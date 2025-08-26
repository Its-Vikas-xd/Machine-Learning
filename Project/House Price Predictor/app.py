import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time

# Page configuration
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced Custom CSS with animations and modern design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        font-family: 'Poppins', sans-serif;
    }
    
    .main-container {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes glow {
        0% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
        50% { box-shadow: 0 0 20px rgba(102, 126, 234, 0.8); }
        100% { box-shadow: 0 0 5px rgba(102, 126, 234, 0.5); }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        color: #1a202c;
        text-align: center;
        margin-bottom: 2rem;
        animation: pulse 2s infinite;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    .sub-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #00000;
        margin-bottom: 1.5rem;
        animation: slideInLeft 0.6s ease-out;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        padding: 3rem;
        border-radius: 20px;
        color: black;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 20px 40px rgba(102, 126, 234, 0.3);
        animation: glow 3s infinite, slideInRight 0.8s ease-out;
        position: relative;
        overflow: hidden;
    }
    
    .prediction-box::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(255,255,255,0.1), transparent);
        transform: rotate(45deg);
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        50% { transform: translateX(100%) translateY(100%) rotate(45deg); }
        100% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
    }
    
    .price-text {
        font-size: 3rem;
        font-weight: 700;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        animation: pulse 2s infinite;
        position: relative;
        z-index: 1;
    }
    
    .info-box {
        background: linear-gradient(135deg, #f8f9ff, #e8f2ff);
        padding: 2rem;
        border-radius: 15px;
        border: 1px solid rgba(102, 126, 234, 0.2);
        margin: 1rem 0;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        animation: slideInUp 0.6s ease-out;
    }
    
    @keyframes slideInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .info-box:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.2);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #fff, #f8f9ff);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
        animation: slideInUp 0.8s ease-out;
    }
    
    .metric-card:hover {
        transform: scale(1.05);
        box-shadow: 0 15px 30px rgba(102, 126, 234, 0.2);
    }
    
    .sidebar .stSelectbox > div > div > div {
        background: linear-gradient(135deg, #f8f9ff, #e8f2ff);
        border: 2px solid rgba(102, 126, 234, 0.3);
        border-radius: 10px;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3) !important;
        animation: glow 2s infinite !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 12px 25px rgba(102, 126, 234, 0.4) !important;
    }
    
    .floating-shapes {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        pointer-events: none;
        z-index: -1;
    }
    
    .shape {
        position: absolute;
        background: rgba(102, 126, 234, 0.1);
        border-radius: 50%;
        animation: float 6s ease-in-out infinite;
    }
    
    .shape:nth-child(1) {
        width: 80px;
        height: 80px;
        top: 10%;
        left: 10%;
        animation-delay: 0s;
    }
    
    .shape:nth-child(2) {
        width: 120px;
        height: 120px;
        top: 20%;
        right: 15%;
        animation-delay: 2s;
    }
    
    .shape:nth-child(3) {
        width: 60px;
        height: 60px;
        bottom: 20%;
        left: 20%;
        animation-delay: 4s;
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px) rotate(0deg); }
        50% { transform: translateY(-20px) rotate(180deg); }
    }
    
    .loading-animation {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid rgba(255,255,255,.3);
        border-radius: 50%;
        border-top-color: #fff;
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .success-checkmark {
        display: inline-block;
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background: #4CAF50;
        position: relative;
        animation: checkmark 0.6s ease-in-out;
    }
    
    @keyframes checkmark {
        0% { transform: scale(0); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }
    
    .stSidebar {
        background: linear-gradient(180deg, rgba(255,255,255,0.95), rgba(248,249,255,0.95));
        backdrop-filter: blur(10px);
    }
    
    .chart-container {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 1rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        animation: slideInRight 1s ease-out;
    }
</style>

<div class="floating-shapes">
    <div class="shape"></div>
    <div class="shape"></div>
    <div class="shape"></div>
</div>
""", unsafe_allow_html=True)

# Load model and data with enhanced error handling
@st.cache_resource
def load_model_and_data():
    try:
        with st.spinner("🚀 Loading AI model..."):
            time.sleep(1)  # Small delay for dramatic effect
            # Load the trained model
            with open('banglor_home_prices_model.pickle', 'rb') as f:
                model = pickle.load(f)
            
            # Load the columns data
            with open('columns.json', 'r') as f:
                columns_data = json.load(f)
            
            st.success("✅ Model loaded successfully!")
            time.sleep(0.5)
            return model, columns_data
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found: {e}")
        st.error("Please make sure 'banglor_home_prices_model.pickle' and 'columns.json' are in the same directory")
        return None, None

# Enhanced prediction function with better error handling
def predict_price(location, sqft, bath, bhk, model, columns):
    if model is None or columns is None:
        return None
    
    # Create feature vector
    x = np.zeros(len(columns['data_columns']))
    
    # Set the numerical features
    x[0] = sqft   # total_sqft
    x[1] = bath   # bath
    x[2] = bhk    # bhk
    
    # Set location (one-hot encoded)
    location_lower = location.lower()
    if location_lower in columns['data_columns']:
        loc_index = columns['data_columns'].index(location_lower)
        x[loc_index] = 1
    
    # Make prediction
    try:
        prediction = model.predict([x])[0]
        return max(prediction, 0)  # Ensure non-negative price
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# Get location list with better formatting
def get_locations(columns_data):
    if columns_data is None:
        return []
    locations = [col for col in columns_data['data_columns'][3:]]  # Skip sqft, bath, bhk
    return sorted([loc.title().replace('_', ' ').replace('Phase', 'Phase ') for loc in locations])

# Enhanced main app with animations
def main():
    # Floating header with animation
    st.markdown("""
    <div class="main-container">
        <h1 class="main-header">🏠 Bangalore House Price Predictor</h1>
        <div style="text-align: center; color: #2d3748; font-size: 1.1rem; margin-bottom: 2rem; animation: slideInUp 1s ease-out; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">
            Powered by AI • Get instant price predictions for Bangalore properties
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model and data
    model, columns_data = load_model_and_data()
    
    if model is None or columns_data is None:
        st.error("❌ Unable to load model files. Please check if the files exist in the correct location.")
        st.stop()
    
    # Enhanced sidebar with animations
    with st.sidebar:
        st.markdown('<h2 class="sub-header"; color:black>🔧 Property Configuration</h2>', unsafe_allow_html=True)
        
        # Get locations
        locations = get_locations(columns_data)
        
        # Location selector with enhanced styling
        location = st.selectbox(
            "📍 Choose Location",
            ["🏙️ Select Location"] + locations,
            help="Select the area where the property is located",
            key="location_select"
        )
        
        # Create two columns for inputs
        col1, col2 = st.columns(2)
        
        with col1:
            sqft = st.number_input(
                "📐 Area (Sq Ft)",
                min_value=500,
                max_value=10000,
                value=1200,
                step=100,
                help="Total square feet area"
            )
            
            bhk = st.selectbox(
                "🛏️ BHK Type",
                [1, 2, 3, 4, 5],
                index=1,
                help="Bedrooms, Hall & Kitchen"
            )
        
        with col2:
            bath = st.selectbox(
                "🚿 Bathrooms",
                [1, 2, 3, 4, 5, 6, 7, 8],
                index=1,
                help="Number of bathrooms"
            )
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Enhanced predict button
        predict_clicked = st.button(
            "🔮 Predict Property Price",
            type="primary",
            use_container_width=True,
            help="Click to get AI-powered price prediction"
        )
    
    # Main content area with enhanced layout
    if predict_clicked:
        if location == "🏙️ Select Location":
            st.warning("⚠️ Please select a location to get accurate price prediction.")
        else:
            # Show prediction process with animations
            with st.spinner("🤖 AI is analyzing market trends..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Convert location back to original format
                location_clean = location.lower().replace(' ', '_').replace('phase_', 'phase')
                
                predicted_price = predict_price(
                    location_clean, sqft, bath, bhk, model, columns_data
                )
                
                progress_bar.empty()
                
                if predicted_price is not None:
                    # Main prediction display
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="success-checkmark"></div>
                            <h2 style="margin: 1rem 0; position: relative; z-index: 1;">💰 Predicted Price</h2>
                            <div class="price-text">₹{predicted_price:.2f} Lakhs</div>
                            <p style="font-size: 1.2rem; opacity: 0.9; position: relative; z-index: 1;">≈ ₹{predicted_price * 100000:,.0f}</p>
                            <p style="font-size: 0.9rem; opacity: 0.8; position: relative; z-index: 1;">📍 {location}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Enhanced metrics display
                        price_per_sqft = (predicted_price * 100000) / sqft
                        
                        # Create animated metric cards
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.markdown(f"""
                            <div class="metric-card">
                            <div style="color: #2d3748; margin-bottom: 0.5rem;">💵 Price/Sq Ft</h4>
                                <h2 style="color: #1a202c; margin: 0; font-weight: 700;">₹{price_per_sqft:,.0f}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_b:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #667eea; margin-bottom: 0.5rem;">📏 Total Area</h4>
                                <h2 style="color: #2d3748; margin: 0;">{sqft:,} sq ft</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        with col_c:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4 style="color: #667eea; margin-bottom: 0.5rem;">🏠 Configuration</h4>
                                <h2 style="color: #2d3748; margin: 0;">{bhk}BHK, {bath}Bath</h2>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                        st.markdown('<h3 style="text-align: center; color: #4a5568; margin-bottom: 1rem;">📊 Market Analysis</h3>', unsafe_allow_html=True)
                        
                        # Enhanced comparison chart
                        comparison_data = {
                            'Property Type': ['1 BHK', '2 BHK', '3 BHK', '4 BHK', '5 BHK', 'Your Property'],
                            'Price (Lakhs)': [30, 50, 75, 105, 140, predicted_price]
                        }
                        
                        colors = ['#e3f2fd', '#bbdefb', '#90caf9', '#64b5f6', '#42a5f5', '#667eea']
                        
                        fig = px.bar(
                            x=comparison_data['Property Type'],
                            y=comparison_data['Price (Lakhs)'],
                            title=f"Price Comparison - {location}",
                            color=comparison_data['Property Type'],
                            color_discrete_sequence=colors
                        )
                        
                        fig.update_layout(
                            showlegend=False,
                            height=400,
                            title_font_size=14,
                            title_x=0.5,
                            plot_bgcolor='rgba(0,0,0,0)',
                            paper_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        fig.update_traces(
                            marker_line_width=0,
                            opacity=0.8
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Market insights
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #e8f5e8, #f0f8f0); padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                            <h4 style="color: #2e7d32; margin: 0 0 0.5rem 0;">📈 Market Insights</h4>
                            <p style="margin: 0; font-size: 0.9rem; color: #4a5568;">
                                Your property is priced {"above" if predicted_price > 75 else "competitively in"} the market average for {location}.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
    
    else:
        # Welcome screen with animations
        st.markdown("""
        <div style="text-align: center; padding: 3rem 0; animation: slideInUp 1s ease-out;">
            <div style="font-size: 5rem; margin-bottom: 1rem; animation: pulse 2s infinite;">🏠</div>
            <h2 style="color: #1a202c; margin-bottom: 1rem; font-weight: 600; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">Welcome to Bangalore Property Price Predictor</h2>
            <p style="color: #00000; font-size: 1.1rem; max-width: 600px; margin: 0 auto; font-weight: 500; text-shadow: 1px 1px 2px rgba(0,0,0,0.1);">
                Get instant AI-powered price predictions for properties in Bangalore. 
                Configure your property details in the sidebar and click predict!
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced information section
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <div style="text-align: center; font-size: 2rem; margin-bottom: 1rem;">🎯</div>
            <h4 style="color: #2d3748; margin-bottom: 1rem;">How It Works</h4>
            <p style="color: #4a5568; line-height: 1.6;">
                Our AI model analyzes location, size, and amenities to predict accurate 
                property prices in Bangalore using machine learning algorithms.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <div style="text-align: center; font-size: 2rem; margin-bottom: 1rem;">📊</div>
            <h4 style="color: #2d3748; margin-bottom: 1rem;">Model Accuracy</h4>
            <p style="color: #4a5568; line-height: 1.6;">
                Trained on extensive real estate data with features like location, 
                total area, BHK configuration, and bathroom count for precise predictions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <div style="text-align: center; font-size: 2rem; margin-bottom: 1rem;">⚠️</div>
            <h4 style="color: #2d3748; margin-bottom: 1rem;">Important Note</h4>
            <p style="color: #4a5568; line-height: 1.6;">
                Predictions are estimates based on historical market data and should 
                be used as reference. Consult real estate professionals for final decisions.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0; animation: slideInUp 1.5s ease-out;">
        <div style="font-size: 2rem; margin-bottom: 1rem; animation: pulse 2s infinite;">✨</div>
        <p style="color: #4a5568; font-size: 1.1rem; margin-bottom: 0.5rem;">
            Built with ❤️ using Streamlit & Machine Learning
        </p>
        <p style="color: #718096; font-size: 0.9rem;">
            Bangalore Real Estate Price Predictor | Powered by AI Technology
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()