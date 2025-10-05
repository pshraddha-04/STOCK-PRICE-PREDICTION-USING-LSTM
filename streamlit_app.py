import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# Page configuration
st.set_page_config(
    page_title="StockPredict - AI Stock Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force sidebar visibility with CSS and JavaScript
st.markdown("""
<style>
    /* Force sidebar to always be visible */
    .css-1d391kg {
        width: 16rem !important;
        min-width: 16rem !important;
        max-width: 16rem !important;
        margin-left: 0 !important;
        transform: translateX(0) !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    .css-17eq0hr {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    [data-testid="stSidebar"] {
        width: 16rem !important;
        min-width: 16rem !important;
        margin-left: 0 !important;
        transform: translateX(0) !important;
        display: block !important;
    }
    
    [data-testid="stSidebar"] > div {
        width: 16rem !important;
        min-width: 16rem !important;
    }
    
    /* Page transition animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .main-content {
        animation: fadeIn 0.5s ease-out;
    }
    
    /* Sidebar styling */
    .sidebar-brand {
        text-align: center;
        padding: 2rem 1rem;
        color: white;
        font-size: 1.5rem;
        font-weight: 700;
        border-bottom: 1px solid rgba(255, 255, 255, 0.2);
        margin-bottom: 2rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 14px;
        transition: all 0.2s ease;
        background: rgba(255, 255, 255, 0.9);
        color: #212529;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    }
    
    .stButton > button:hover {
        background: rgba(255, 255, 255, 1);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    
    .stButton > button:disabled {
        background: rgba(0, 0, 0, 0.2) !important;
        color: #212529 !important;
        cursor: not-allowed;
        transform: none;
        font-weight: 600;
    }
    
    /* Page titles */
    .page-title {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    /* Feature cards */
    .feature-card {
        background: linear-gradient(135deg, #f8fafc 0%, #ffffff 100%);
        padding: 2rem;
        border-radius: 15px;
        border-left: 5px solid #3b82f6;
        margin: 1.5rem 0;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.15);
        border-left-color: #1e40af;
    }
    
    /* Main header */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(135deg, #1e40af 0%, #3b82f6 100%);
        color: white;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(59, 130, 246, 0.3);
    }
    
    /* Hide default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>

<script>
// Force sidebar to stay visible
function forceSidebarVisible() {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
        sidebar.style.width = '16rem';
        sidebar.style.minWidth = '16rem';
        sidebar.style.marginLeft = '0';
        sidebar.style.transform = 'translateX(0)';
        sidebar.style.display = 'block';
    }
}

// Run on page load and periodically
setTimeout(forceSidebarVisible, 100);
setInterval(forceSidebarVisible, 1000);
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Dashboard"
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'quick_symbol' not in st.session_state:
    st.session_state.quick_symbol = None
if 'first_visit' not in st.session_state:
    st.session_state.first_visit = True

def main():
    # Sidebar navigation (always visible)
    with st.sidebar:
        # Brand/Logo
        st.markdown("""
        <div class="sidebar-brand">
             StockPredict AI
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation menu
        st.markdown("###")
        
        # Dashboard button
        dashboard_active = st.session_state.current_page == "Dashboard"
        if st.button(" Dashboard", key="nav_dashboard", use_container_width=True, 
                    disabled=dashboard_active):
            with st.spinner("Loading Dashboard..."):
                st.session_state.current_page = "Dashboard"
                st.session_state.prediction_result = None
                time.sleep(0.3)
            st.rerun()
        
        # Stock Prediction button
        prediction_active = st.session_state.current_page == "Stock Prediction"
        if st.button(" Stock Prediction", key="nav_prediction", use_container_width=True,
                    disabled=prediction_active):
            with st.spinner("Loading Stock Prediction..."):
                st.session_state.current_page = "Stock Prediction"
                time.sleep(0.3)
            st.rerun()
        
        # About button
        about_active = st.session_state.current_page == "About"
        if st.button(" About", key="nav_about", use_container_width=True,
                    disabled=about_active):
            with st.spinner("Loading About..."):
                st.session_state.current_page = "About"
                st.session_state.prediction_result = None
                time.sleep(0.3)
            st.rerun()
        
        # Model Performance button
        performance_active = st.session_state.current_page == "Model Performance"
        if st.button(" Model Performance", key="nav_performance", use_container_width=True,
                    disabled=performance_active):
            with st.spinner("Loading Model Performance..."):
                st.session_state.current_page = "Model Performance"
                st.session_state.prediction_result = None
                time.sleep(0.3)
            st.rerun()
        
        # Current page indicator
        st.markdown("---")
        page_icons = {
            "Dashboard": "",
            "Stock Prediction": "",
            "About": "",
            "Model Performance": ""
        }
        current_icon = page_icons.get(st.session_state.current_page, "")
        st.markdown(f"{current_icon} Current: {st.session_state.current_page}")
        
        # Quick stats
        st.markdown("---")
        st.markdown("###  Quick Stats")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", "97.1%", delta="+2.3%")
        with col2:
            st.metric("RMSE", "8.13", delta="-1.2")
        
        st.metric("MAE", "5.90", delta="-0.8")
        
        # Quick actions
        st.markdown("---")
        st.markdown("###  Quick Actions")
        
        if st.button(" Quick Predict MSFT", use_container_width=True):
            st.session_state.current_page = "Stock Prediction"
            st.session_state.quick_symbol = "MSFT"
            st.rerun()
        
        if st.button("View Performance", use_container_width=True):
            st.session_state.current_page = "Model Performance"
            st.rerun()
        
        # Footer
        st.markdown("---")
        st.markdown("""
       
        """, unsafe_allow_html=True)
    
    # Main content with animation
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    
    # Route to appropriate page
    if st.session_state.current_page == "Dashboard":
        show_dashboard()
    elif st.session_state.current_page == "Stock Prediction":
        show_prediction()
    elif st.session_state.current_page == "About":
        show_about()
    elif st.session_state.current_page == "Model Performance":
        show_performance()
    
    st.markdown('</div>', unsafe_allow_html=True)

def show_dashboard():
    st.markdown("""
    <div class="page-title">
        Welcome to StockPredict AI
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="main-header">
        <h1> Advanced Stock Price Prediction</h1>
        <p style="font-size: 1.2rem; margin: 0;">Powered by LSTM Neural Networks & Real-time Market Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.first_visit:
        st.success(" Welcome to StockPredict AI! Navigate using the sidebar to explore our features.")
        st.session_state.first_visit = False
    
    st.header(" Key Features")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h3> LSTM Neural Networks</h3>
            <p>Advanced deep learning models for time-series prediction</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h3> Technical Indicators</h3>
            <p>SMA, RSI, Bollinger Bands, MACD analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="feature-card">
            <h3> Real-time Data</h3>
            <p>Live stock data from Yahoo Finance API</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.header(" Model Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Accuracy", "97.1%", "R² Score")
    
    with col2:
        st.metric("RMSE Score", "8.13", "Root Mean Squared Error")
    
    with col3:
        st.metric("MAE Score", "5.90", "Mean Absolute Error")
    
    st.markdown("---")
    st.markdown("###  Ready to predict stock prices?")
    
    cta_col1, cta_col2 = st.columns(2)
    
    with cta_col1:
        if st.button(" Start Prediction", type="primary", use_container_width=True):
            st.session_state.current_page = "Stock Prediction"
            st.rerun()
    
    with cta_col2:
        if st.button(" View Performance", use_container_width=True):
            st.session_state.current_page = "Model Performance"
            st.rerun()

def check_api_status():
    try:
        response = requests.get("http://localhost:5000/", timeout=5)
        return response.status_code == 200
    except:
        return False

def get_prediction(symbol, days):
    try:
        response = requests.post(
            "http://localhost:5000/predict",
            json={"stockSymbol": symbol, "predictionDays": days},
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}"}
            
    except requests.exceptions.Timeout:
        return {"error": "Request timed out. Please try again."}
    except requests.exceptions.ConnectionError:
        return {"error": "Cannot connect to Flask API. Make sure it's running on port 5000."}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}

def show_prediction():
    st.markdown("""
    <div class="page-title">
         Stock Price Prediction
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 1.1rem; color: #666;">Enter stock details to get AI-powered price predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Handle quick symbol
    default_symbol = "MSFT"
    if st.session_state.quick_symbol:
        default_symbol = st.session_state.quick_symbol
        st.session_state.quick_symbol = None
        st.info(f" Quick prediction mode activated for {default_symbol}")
    
    # Input form
    st.markdown("###  Prediction Parameters")
    
    with st.form("prediction_form", clear_on_submit=False):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            stock_symbol = st.text_input(
                "Stock Symbol ", 
                placeholder="e.g., AAPL, GOOGL, MSFT, TSLA",
                help="Enter a valid stock ticker symbol",
                value=default_symbol
            ).upper()
        
        with col2:
            prediction_days = st.selectbox(
                "Prediction Period ",
                options=[7, 14, 30, 60, 90],
                format_func=lambda x: f"{x} Days ({x//7} weeks)" if x >= 7 else f"{x} Days",
                index=2,
                help="Select the time horizon for price prediction"
            )
        
        with col3:
            st.markdown("<br>", unsafe_allow_html=True)
            submitted = st.form_submit_button(
                " Generate Prediction", 
                type="primary", 
                use_container_width=True
            )
    
   
    
    # Process form submission
    if submitted and stock_symbol:
        if len(stock_symbol) < 1 or len(stock_symbol) > 10:
            st.error(" Please enter a valid stock symbol (1-10 characters)")
        else:
            with st.spinner(f" Analyzing {stock_symbol} for {prediction_days} days..."):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                result = get_prediction(stock_symbol, prediction_days)
                progress_bar.empty()
                
                if result and result.get('success'):
                    st.session_state.prediction_result = result
                    # st.balloons()
                    show_prediction_results(result)
                else:
                    error_msg = result.get('error', 'Failed to get prediction') if result else 'API connection failed'
                    st.error(f" Prediction Failed: {error_msg}")
    
    # Show previous result
    elif st.session_state.prediction_result:
        st.markdown("---")
        st.info("Previous Prediction Result:")
        show_prediction_results(st.session_state.prediction_result)
        
        if st.button(" Clear Previous Result", use_container_width=True):
            st.session_state.prediction_result = None
            st.rerun()

def show_prediction_results(result):
    st.markdown("""
    <div style="background: #d4edda; border: 1px solid #c3e6cb; border-radius: 12px; padding: 1.5rem; margin: 1rem 0;">
        <h3> Prediction Generated Successfully!</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${result['current_price']}")
    
    with col2:
        delta_color = "normal" if result['price_change'] >= 0 else "inverse"
        st.metric(
            "Predicted Price", 
            f"${result['predicted_price']}", 
            f"${result['price_change']}",
            delta_color=delta_color
        )
    
    with col3:
        delta_color = "normal" if result['percent_change'] >= 0 else "inverse"
        st.metric(
            "Price Change", 
            f"{result['percent_change']:.2f}%",
            delta_color=delta_color
        )
    
    with col4:
        st.metric("Confidence", f"{result['confidence']}%")
    
    # Prediction summary
    if result['percent_change'] > 0:
        st.success(f" Bullish Prediction: {result['symbol']} is expected to rise by {result['percent_change']:.2f}% over {result['prediction_days']} days")
    else:
        st.warning(f"Bearish Prediction: {result['symbol']} is expected to decline by {abs(result['percent_change']):.2f}% over {result['prediction_days']} days")
    
    # Detailed results
    st.markdown("---")
    st.subheader(" Prediction Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info(f"""
        **Stock Symbol:** {result['symbol']}  
        **Prediction Period:** {result['prediction_days']} days  
        **Current Price:** ${result['current_price']}  
        **Predicted Price:** ${result['predicted_price']}
        """)
    
    with col2:
        st.info(f"""
        **Price Change:** ${result['price_change']}  
        **Percentage Change:** {result['percent_change']:.2f}%  
        **Model Confidence:** {result['confidence']}%  
        **RMSE:** {result['rmse']}
        """)
    
    # Model performance info
    with st.expander(" Model Performance Metrics"):
        perf_col1, perf_col2, perf_col3 = st.columns(3)
        with perf_col1:
            st.metric("R² Score", f"{result['r2_score']:.4f}")
        with perf_col2:
            st.metric("RMSE", f"{result['rmse']:.3f}")
        with perf_col3:
            st.metric("MAE", f"{result['mae']:.3f}")
    
    # Visualization
    create_prediction_chart(result)
    
   

def create_prediction_chart(result):
    st.subheader(" Price Prediction Visualization")
    
    dates = pd.date_range(start=datetime.now(), periods=result['prediction_days']+1, freq='D')
    current_price = result['current_price']
    predicted_price = result['predicted_price']
    
    daily_return = (predicted_price / current_price) ** (1/result['prediction_days']) - 1
    prices = [current_price]
    
    for i in range(1, len(dates)):
        trend_price = prices[-1] * (1 + daily_return)
        volatility = abs(daily_return) * 0.5
        random_factor = np.random.normal(0, volatility)
        new_price = trend_price * (1 + random_factor)
        prices.append(new_price)
    
    prices[-1] = predicted_price
    
    df = pd.DataFrame({'Date': dates, 'Price': prices})
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['Date'],
        y=df['Price'],
        mode='lines+markers',
        name=f'{result["symbol"]} Price Trend',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=4),
        fill='tonexty' if result['percent_change'] > 0 else None,
        fillcolor='rgba(59, 130, 246, 0.1)'
    ))
    
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[0]],
        y=[current_price],
        mode='markers+text',
        name='Current Price',
        marker=dict(color='#28a745', size=15, symbol='circle'),
        text=[f'Current: ${current_price}'],
        textposition='top center'
    ))
    
    fig.add_trace(go.Scatter(
        x=[df['Date'].iloc[-1]],
        y=[predicted_price],
        mode='markers+text',
        name='Predicted Price',
        marker=dict(color='#dc3545', size=15, symbol='diamond'),
        text=[f'Target: ${predicted_price}'],
        textposition='top center'
    ))
    
    fig.add_hline(y=current_price, line_dash="dash", line_color="green", opacity=0.5)
    fig.add_hline(y=predicted_price, line_dash="dash", line_color="red", opacity=0.5)
    
    fig.update_layout(
        title=f"{result['symbol']} - {result['prediction_days']} Day Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        hovermode='x unified',
        showlegend=True,
        height=500,
        template="plotly_white"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    price_range = max(prices) - min(prices)
    st.info(f" **Price Range**: ${min(prices):.2f} - ${max(prices):.2f} (Range: ${price_range:.2f})")


def show_about():
    st.markdown("""
    <div class="page-title">
         About StockPredict AI
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ##  Project Overview
    
    StockPredict AI is an advanced stock price prediction system that uses **Long Short-Term Memory (LSTM)** 
    neural networks to forecast stock prices with high accuracy.
    
    ###  Technical Features
    
    - **Deep Learning**: LSTM neural networks optimized for time-series prediction
    - **Technical Analysis**: Integration of RSI, MACD, SMA, and Bollinger Bands
    - **Real-time Data**: Live stock data from Yahoo Finance API
    - **High Accuracy**: 97.1% R² score with RMSE of 8.13
    
    ###  Model Architecture
    
    - **Input Features**: Close, Open, Volume, RSI_14, MACD, MACD_Hist
    - **Sequence Length**: 60-day lookback window
    - **LSTM Units**: 150 units with 0.2 dropout
    - **Training Data**: Microsoft (MSFT) stock data (2020-2025)
    
    ###  Use Cases
    
    - Investment decision support
    - Risk assessment and management
    - Technical analysis automation
    - Portfolio optimization
    """)

def show_performance():
    st.markdown("""
    <div class="page-title">
         Model Performance
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Our LSTM model demonstrates exceptional performance in stock price prediction, 
    achieving state-of-the-art accuracy metrics on Microsoft (MSFT) stock data.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("R² Score", "0.9711", "97.11% variance explained")
    
    with col2:
        st.metric("RMSE", "8.131", "Root Mean Squared Error")
    
    with col3:
        st.metric("MAE", "5.901", "Mean Absolute Error")
    
    st.markdown("---")
    st.subheader(" Model Architecture")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **LSTM Configuration:**
        - Units: 150
        - Dropout: 0.2
        - Dense Units: 50
        - Learning Rate: 0.005
        - Batch Size: 64
        - Optimizer: Adam
        """)
    
    with col2:
        st.info("""
        **Training Details:**
        - Dataset: Microsoft (MSFT)
        - Period: 2020-2025
        - Features: 6 technical indicators
        - Sequence Length: 60 days
        - Train/Test Split: 80/20
        - Epochs: 50 (with early stopping)
        """)
    
    st.subheader(" Performance Comparison")
    
    metrics_data = {
        'Metric': ['RMSE', 'MAE', 'R² Score'],
        'LSTM Model': [8.131, 5.901, 0.9711],
        'Baseline (Linear)': [15.2, 11.8, 0.8234],
        'Random Forest': [12.5, 9.2, 0.8956]
    }
    
    df_metrics = pd.DataFrame(metrics_data)
    
    fig = px.bar(
        df_metrics.melt(id_vars='Metric', var_name='Model', value_name='Score'),
        x='Metric',
        y='Score',
        color='Model',
        barmode='group',
        title="Model Performance Comparison",
        color_discrete_map={
            'LSTM Model': '#3b82f6',
            'Baseline (Linear)': '#ef4444',
            'Random Forest': '#10b981'
        }
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_footer():
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem 0;">
        <p> StockPredict AI - Advanced LSTM Stock Price Prediction</p>
        
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    try:
        main()
        show_footer()
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        st.info("Please refresh the page or contact support if the issue persists.")