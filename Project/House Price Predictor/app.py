import streamlit as st
import pickle
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Bangalore House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #A23B72;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .price-text {
        font-size: 2.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .info-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #2E86AB;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load model and data
@st.cache_resource
def load_model_and_data():
    try:
        # Load the trained model
        with open('banglor_home_prices_model.pickle', 'rb') as f:
            model = pickle.load(f)
        
        # Load the columns data
        with open('columns.json', 'r') as f:
            columns_data = json.load(f)
        
        return model, columns_data
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.error("Please make sure 'banglor_home_prices_model.pickle' and 'columns.json' are in the same directory")
        return None, None

# Prediction function
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

# Get location list (excluding numerical features)
def get_locations(columns_data):
    if columns_data is None:
        return []
    locations = [col for col in columns_data['data_columns'][3:]]  # Skip sqft, bath, bhk
    return [loc.title().replace('_', ' ') for loc in locations]

# Main app
def main():
    st.markdown('<h1 class="main-header">🏠 Bangalore House Price Predictor</h1>', unsafe_allow_html=True)
    
    # Load model and data
    model, columns_data = load_model_and_data()
    
    if model is None or columns_data is None:
        st.error("Unable to load model files. Please check if the files exist in the correct location.")
        st.stop()
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="sub-header">🔧 Property Details</h2>', unsafe_allow_html=True)
    
    # Get locations
    locations = get_locations(columns_data)
    
    # Input fields
    location = st.sidebar.selectbox(
        "📍 Select Location",
        ["Select Location"] + locations,
        help="Choose the location of the property"
    )
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        sqft = st.number_input(
            "📐 Total Sq Ft",
            min_value=500,
            max_value=10000,
            value=1200,
            step=100,
            help="Total square feet area of the property"
        )
        
        bhk = st.selectbox(
            "🛏️ BHK",
            [1, 2, 3, 4, 5],
            index=1,
            help="Number of bedrooms, hall, and kitchen"
        )
    
    with col2:
        bath = st.selectbox(
            "🚿 Bathrooms",
            [1, 2, 3, 4, 5, 6],
            index=1,
            help="Number of bathrooms"
        )
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True):
            if location == "Select Location":
                st.warning("Please select a location to get price prediction.")
            else:
                with st.spinner("Calculating price prediction..."):
                    # Convert location back to original format for prediction
                    location_original = location.lower().replace(' ', '_')
                    
                    predicted_price = predict_price(
                        location_original, sqft, bath, bhk, model, columns_data
                    )
                    
                    if predicted_price is not None:
                        st.markdown(f"""
                        <div class="prediction-box">
                            <h2>💰 Predicted Price</h2>
                            <div class="price-text">₹{predicted_price:.2f} Lakhs</div>
                            <p>≈ ₹{predicted_price * 100000:,.0f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Price per sq ft
                        price_per_sqft = (predicted_price * 100000) / sqft
                        
                        # Display additional metrics
                        col_a, col_b, col_c = st.columns(3)
                        
                        with col_a:
                            st.metric("Price per Sq Ft", f"₹{price_per_sqft:,.0f}")
                        
                        with col_b:
                            st.metric("Total Area", f"{sqft:,} sq ft")
                        
                        with col_c:
                            st.metric("Configuration", f"{bhk} BHK, {bath} Bath")
    
    with col2:
        st.markdown('<h3 class="sub-header">📊 Property Analysis</h3>', unsafe_allow_html=True)
        
        # Create a sample comparison chart if prediction was made
        if 'predicted_price' in locals() and predicted_price is not None:
            # Sample data for comparison (you can replace this with actual market data)
            comparison_data = {
                'Property Type': ['1 BHK', '2 BHK', '3 BHK', '4 BHK', 'Your Property'],
                'Avg Price (Lakhs)': [45, 65, 85, 120, predicted_price],
                'Color': ['lightblue', 'lightblue', 'lightblue', 'lightblue', 'red']
            }
            
            fig = px.bar(
                x=comparison_data['Property Type'],
                y=comparison_data['Avg Price (Lakhs)'],
                color=comparison_data['Color'],
                title=f"Price Comparison in {location}",
                labels={'x': 'Property Type', 'y': 'Price (Lakhs)'},
                color_discrete_map={'lightblue': 'lightblue', 'red': 'red'}
            )
            fig.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig, use_container_width=True)
    
    # Information section
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🎯 How it works</h4>
            <p>This model uses machine learning to predict house prices based on location, size, and amenities in Bangalore.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📈 Model Info</h4>
            <p>Trained on real estate data with features like location, total area, BHK, and bathrooms.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>⚠️ Disclaimer</h4>
            <p>Predictions are estimates based on historical data and should not be considered as professional advice.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; margin-top: 2rem;">
            <p>Built with ❤️ using Streamlit | Bangalore Real Estate Price Predictor</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()