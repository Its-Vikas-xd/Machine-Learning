import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer
import time
import plotly.graph_objects as go
from datetime import datetime

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)


# Page configuration
st.set_page_config(
    page_title="Spam Detector Pro",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }
    
    .result-box {
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        font-size: 1.5rem;
        font-weight: bold;
    }
    
    .spam-result {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        color: white;
        border: 2px solid #e73827;
        box-shadow: 0 10px 30px rgba(255, 65, 108, 0.4);
        animation: pulse 2s infinite;
    }
    
    .safe-result {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        border: 2px solid #0d7377;
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.4);
        animation: glow 2s ease-in-out infinite alternate;
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 10px 30px rgba(255, 65, 108, 0.4); }
        50% { box-shadow: 0 15px 40px rgba(255, 65, 108, 0.6); }
        100% { box-shadow: 0 10px 30px rgba(255, 65, 108, 0.4); }
    }
    
    @keyframes glow {
        from { box-shadow: 0 10px 30px rgba(17, 153, 142, 0.4); }
        to { box-shadow: 0 15px 40px rgba(17, 153, 142, 0.6); }
    }
    
    .info-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #4c63d2;
        margin: 1rem 0;
        color: white;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    
    .info-card-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        border-left: 5px solid #0d7377;
        box-shadow: 0 8px 25px rgba(17, 153, 142, 0.3);
    }
    
    .info-card-spam {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        border-left: 5px solid #e73827;
        box-shadow: 0 8px 25px rgba(255, 65, 108, 0.3);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.2);
        text-align: center;
        margin: 1rem 0;
        color: white;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .metric-card-spam {
        background: linear-gradient(135deg, #ff416c 0%, #ff4b2b 100%);
        box-shadow: 0 10px 30px rgba(255, 65, 108, 0.2);
    }
    
    .metric-card-safe {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        box-shadow: 0 10px 30px rgba(17, 153, 142, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load models with error handling
@st.cache_resource
def load_models():
    try:
        tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
        model = pickle.load(open('model.pkl', 'rb'))
        return tfidf, model
    except FileNotFoundError:
        st.error("⚠️ Model files not found! Please ensure 'vectorizer.pkl' and 'model.pkl' are in the same directory.")
        return None, None

tfidf, model = load_models()

# Header
st.markdown('<h1 class="main-header">🛡️ Spam Detector Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered spam detection for emails and SMS messages</p>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Dashboard")
    st.markdown("---")
    
    # Statistics
    total_predictions = len(st.session_state.prediction_history)
    if total_predictions > 0:
        spam_count = sum(1 for pred in st.session_state.prediction_history if pred['result'] == 'Spam')
        spam_percentage = (spam_count / total_predictions) * 100
        
        st.markdown(f"""
        <div class="metric-card">
            <h3>🔢 Total Predictions</h3>
            <h2 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{total_predictions}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card metric-card-spam">
            <h3>🚨 Spam Detected</h3>
            <h2 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{spam_count}</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f"""
        <div class="metric-card metric-card-safe">
            <h3>📈 Spam Rate</h3>
            <h2 style="color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);">{spam_percentage:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("No predictions made yet. Start analyzing messages!")
    
    st.markdown("---")
    
    # Information section
    with st.expander("ℹ️ How it works"):
        st.markdown("""
        **Our AI model uses:**
        - Natural Language Processing
        - TF-IDF Vectorization
        - Machine Learning Classification
        - Text preprocessing and cleaning
        
        **Features:**
        - Real-time spam detection
        - Confidence scoring
        - Prediction history
        - Performance analytics
        """)
    
    # Clear history button
    if st.button("🗑️ Clear History", type="secondary"):
        st.session_state.prediction_history = []
        st.success("History cleared!")

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### 📝 Enter Your Message")
    
    # Text input options
    input_method = st.radio(
        "Choose input method:",
        ["Type message", "Upload text file"],
        horizontal=True
    )
    
    input_sms = ""
    if input_method == "Type message":
        input_sms = st.text_area(
            "Message content:",
            placeholder="Enter your email or SMS message here...",
            height=150,
            help="Paste any email or SMS content to check if it's spam"
        )
    else:
        uploaded_file = st.file_uploader(
            "Upload a text file",
            type=['txt'],
            help="Upload a .txt file containing the message to analyze"
        )
        if uploaded_file is not None:
            input_sms = str(uploaded_file.read(), "utf-8")
            st.text_area("File content:", value=input_sms, height=150, disabled=True)

with col2:
    st.markdown("### 🎯 Quick Examples")
    
    examples = [
        "Congratulations! You've won $1000. Click here to claim your prize!",
        "Hi, this is a reminder about your appointment tomorrow at 3 PM.",
        "URGENT: Your account will be suspended. Verify now!",
        "Thanks for the meeting today. Let's catch up next week."
    ]
    
    for i, example in enumerate(examples, 1):
        if st.button(f"Example {i}", key=f"example_{i}", help=example[:50] + "..."):
            st.session_state.selected_example = example
            input_sms = example

# Check if example was selected
if 'selected_example' in st.session_state:
    input_sms = st.session_state.selected_example

# Text preprocessing function
ps = PorterStemmer()

def transform_text(text):
    """Enhanced text preprocessing with better error handling"""
    try:
        text = text.lower()
        text = nltk.word_tokenize(text)
        
        # Remove non-alphanumeric
        y = [i for i in text if i.isalnum()]
        
        # Remove stopwords and punctuation
        y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
        
        # Stemming
        y = [ps.stem(i) for i in y]
        
        return " ".join(y)
    except Exception as e:
        st.error(f"Error processing text: {str(e)}")
        return ""

# Prediction section
st.markdown("---")

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    predict_button = st.button(
        "🔍 Analyze Message", 
        type="primary", 
        use_container_width=True,
        disabled=(not input_sms.strip() or tfidf is None or model is None)
    )

if predict_button and input_sms.strip() and tfidf is not None and model is not None:
    with st.spinner("🤖 Analyzing message..."):
        # Simulate processing time for better UX
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        # Text preprocessing
        transformed_sms = transform_text(input_sms)
        
        if transformed_sms:
            # Vectorization and prediction
            vector_input = tfidf.transform([transformed_sms])
            result = model.predict(vector_input)[0]
            
            # Get prediction probabilities if available
            try:
                probabilities = model.predict_proba(vector_input)[0]
                confidence = max(probabilities) * 100
            except:
                confidence = 85  # Default confidence if probabilities not available
            
            # Display result
            if result == 1:
                st.markdown(f"""
                <div class="result-box spam-result">
                    🚨 SPAM DETECTED
                    <br>
                    <small>Confidence: {confidence:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
                result_text = "Spam"
                result_color = "#f44336"
            else:
                st.markdown(f"""
                <div class="result-box safe-result">
                    ✅ MESSAGE IS SAFE
                    <br>
                    <small>Confidence: {confidence:.1f}%</small>
                </div>
                """, unsafe_allow_html=True)
                result_text = "Not Spam"
                result_color = "#4caf50"
            
            # Add to history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'message': input_sms[:100] + "..." if len(input_sms) > 100 else input_sms,
                'result': result_text,
                'confidence': confidence
            })
            
            # Show confidence visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = confidence,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': result_color},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"},
                        {'range': [80, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("⚠️ Could not process the message. Please try again with a different text.")

# Recent predictions history
if st.session_state.prediction_history:
    st.markdown("---")
    st.markdown("## 📈 Recent Predictions")
    
    # Show last 5 predictions
    recent_predictions = st.session_state.prediction_history[-5:][::-1]
    
    for pred in recent_predictions:
        result_emoji = "🚨" if pred['result'] == 'Spam' else "✅"
        card_class = "info-card info-card-spam" if pred['result'] == 'Spam' else "info-card info-card-safe"
        
        st.markdown(f"""
        <div class="{card_class}">
            <strong>{result_emoji} {pred['result']}</strong> - {pred['confidence']:.1f}% confidence
            <br>
            <small style="opacity: 0.9;">🕒 {pred['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}</small>
            <br>
            <em style="opacity: 0.9;">"{pred['message']}"</em>
        </div>
        """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🛡️ <b>Spam Detector Pro</b> | Powered by AI | Keep your messages safe</p>
        <p><small>Built with Streamlit • Machine Learning • NLP</small></p>
        <p>
            🌐 <a href="https://vikas-portfolio-chi.vercel.app/" target="_blank">Portfolio</a> | 
            💻 <a href="https://github.com/Its-Vikas-xd" target="_blank">GitHub</a> | 
            🔗 <a href="https://www.linkedin.com/in/vikas-sharma-493115361/" target="_blank">LinkedIn</a> | 
            ✖️ <a href="https://x.com/ItsVikasXd" target="_blank">X (Twitter)</a>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)
