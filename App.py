import streamlit as st  # type: ignore
import pandas as pd  # type: ignore
import plotly.graph_objects as go  # type: ignore
import yfinance as yf  # type: ignore
from phi.agent.agent import Agent  # type: ignore
from phi.model.groq import Groq  # type: ignore
from phi.tools.yfinance import YFinanceTools  # type: ignore
from phi.tools.duckduckgo import DuckDuckGo  # type: ignore
from phi.tools.googlesearch import GoogleSearch  # type: ignore
import os

# API Key: Replace with your actual Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_api_key_here")

# Common stock symbols
COMMON_STOCKS = {
    'NVIDIA': 'NVDA', 'APPLE': 'AAPL', 'GOOGLE': 'GOOGL', 'MICROSOFT': 'MSFT',
    'TESLA': 'TSLA', 'AMAZON': 'AMZN', 'META': 'META', 'NETFLIX': 'NFLX',
    'TCS': 'TCS.NS', 'RELIANCE': 'RELIANCE.NS', 'INFOSYS': 'INFY.NS',
    'WIPRO': 'WIPRO.NS', 'HDFC': 'HDFCBANK.NS', 'TATAMOTORS': 'TATAMOTORS.NS',
    'ICICIBANK': 'ICICIBANK.NS', 'SBIN': 'SBIN.NS'
}

# Streamlit app settings
st.set_page_config(
    page_title="TOP STOCKS FULL ANALYSIS & RECOMMENDATION SYSTEM PREDICTION AI Agents",
    page_icon="📈",
    layout="wide"
)
st.sidebar.title("Stock Market Analysis")

# Add stock png to sidebar
stock_png_path = "stock.png"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image("stock.png", use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("stock.png. Please check the file path.")

# Add market png to sidebar
market_png_path = "market.png"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image("market.png", use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("market.png. Please check the file path.")

st.sidebar.title("Developer: Abhishek Kumar")

# Add my jpg to sidebar
pic_jpg_path = "pic.jpg"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image("pic.jpg", use_container_width=True)
except FileNotFoundError:
    st.sidebar.warning("pic.jpg. Please check the file path.")


def initialize_agents():
    """Initialize AI agents with proper API key and tools."""
    if not st.session_state.get('agents_initialized', False):
        try:
            # Initialize the web search agent
            st.session_state.web_agent = Agent(
                name="Web Search Agent",
                role="Search the web for information",
                model=Groq(api_key=GROQ_API_KEY, model_id="llama-3.3-70b-versatile"),
                tools=[GoogleSearch(fixed_max_results=5), DuckDuckGo(fixed_max_results=5)]
            )
            # Initialize the financial AI agent
            st.session_state.finance_agent = Agent(
                name="Financial AI Agent",
                role="Providing financial insights",
                model=Groq(api_key=GROQ_API_KEY, model_id="llama-3.3-70b-versatile"),
                tools=[YFinanceTools()]
            )
            st.session_state.agents_initialized = True
        except Exception as e:
            st.error(f"Agent initialization error: {str(e)}")
            st.session_state.agents_initialized = False


def get_stock_data(symbol, start_date, end_date):
    """Fetch stock data using yfinance."""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        hist = stock.history(start=start_date, end=end_date)
        return info, hist
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None, None


def create_price_chart(hist_data, symbol):
    """Generate a candlestick and volume chart for the stock."""
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=hist_data.index, open=hist_data['Open'],
        high=hist_data['High'], low=hist_data['Low'],
        close=hist_data['Close'], name='OHLC'
    ))
    fig.add_trace(go.Bar(
        x=hist_data.index, y=hist_data['Volume'],
        name='Volume', marker_color='rgba(50, 150, 255, 0.5)'
    ))
    fig.update_layout(
        title=f'{symbol} Price Movement',
        template='plotly_white',
        xaxis_rangeslider_visible=False,
        height=500
    )
    return fig


def feedback_section():
    """Handle feedback submission and display a thank-you message."""
    st.markdown("### Your Feedback")

    # Initialize session state for feedback
    if "feedback_submitted" not in st.session_state:
        st.session_state.feedback_submitted = False
        st.session_state.feedback = None

    # Display feedback form only if feedback is not submitted
    if not st.session_state.feedback_submitted:
        feedback = st.radio(
            "Was this analysis helpful?", options=["Yes", "No", "Needs Improvement"]
        )
        submit_button = st.button("Submit Feedback", key="submit_feedback")
        
        # If submit button is clicked, update session state
        if submit_button:
            st.session_state.feedback_submitted = True
            st.session_state.feedback = feedback
            st.rerun()  # This forces a rerun, preserving the feedback state
    else:
        # Show a thank-you message based on feedback
        feedback = st.session_state.feedback
        if feedback == "Yes":
            st.success("""
                Thank you for your positive feedback! 😊  
                We're thrilled you found the analysis helpful.  
                Your support inspires us to do even better.  
                Happy investing, and have a fantastic day! 🚀
            """)
        elif feedback == "No":
            st.error("""
                We're sorry the analysis didn't meet your expectations. 😔  
                Your feedback helps us improve.  
                We'll strive to make things better for you! 🙏  
                Thank you for sharing your thoughts! 💪
            """)
        else:  # "Needs Improvement"
            st.warning("""
                Thank you for your feedback! 📝  
                Your suggestions are invaluable to us.  
                We're working to enhance your experience.  
                Expect better updates soon! 💛
            """)

        # Option to reset feedback submission
        reset_button = st.button("Submit Another Feedback", key="reset_feedback")
        if reset_button:
            st.session_state.feedback_submitted = False  # Reset feedback state


def main():
    """Main function to run the Streamlit app."""
    st.title("TOP STOCKS FULL ANALYSIS & RECOMMENDATION SYSTEM PREDICTION AI Agents")

    # Dropdown for stock selection
    stock_selection = st.selectbox("Select a Company", options=[""] + list(COMMON_STOCKS.keys()), help="Select a stock from the dropdown or type manually")
    stock_input = stock_selection if stock_selection else st.text_input("Enter Company Name", help="e.g., APPLE, TCS")

    # Date range input
    start_date, end_date = st.date_input("Select Date Range for Analysis", value=[pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01")])

    if st.button("Analyze", use_container_width=True):
        if not stock_input:
            st.error("Please select or enter a stock name")
            return

        symbol = COMMON_STOCKS.get(stock_input.upper()) or stock_input

        if not st.session_state.get("agents_initialized", False):
            initialize_agents()

        with st.spinner("Analyzing..."):
            info, hist = get_stock_data(symbol, start_date, end_date)

            if info and hist is not None:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(label="Current Price", value=f"${info.get('currentPrice', 'N/A')}")
                with col2:
                    st.metric(label="Forward P/E", value=f"{info.get('forwardPE', 'N/A')}")
                with col3:
                    st.metric(label="Recommendation", value=info.get('recommendationKey', 'N/A').title())

                st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
                st.plotly_chart(create_price_chart(hist, symbol), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

                if 'longBusinessSummary' in info:
                    st.markdown("<div class='card'>", unsafe_allow_html=True)
                    st.markdown("### Company Overview")
                    st.write(info['longBusinessSummary'])
                    st.markdown("</div>", unsafe_allow_html=True)

                # Feedback Section
                feedback_section()


if __name__ == "__main__":
    main()
