import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import asyncio
from typing import Dict, List, Any, Optional
import logging
from dataclasses import dataclass
import requests
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langsmith import Client
from langgraph.graph import END, StateGraph
from langchain.tools import BaseTool
from langchain.callbacks.tracers import LangChainTracer
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Initialize LangSmith (will be conditionally enabled)
os.environ["LANGCHAIN_PROJECT"] = "financial-analysis-agent"

# New imports for ranking functionality
from typing import TypedDict, Annotated
import math
import time

def show_thinking_animation():
    """Show a thinking animation with different states"""
    
    # Create thinking animation
    thinking_container = st.container()
    with thinking_container:
        st.markdown("🤔 **FinBot is thinking...**")
        
        # Create animated dots
        dots = st.empty()
        for i in range(3):
            dots.markdown("." * (i + 1))
            time.sleep(0.3)
        
        dots.empty()
    
    return thinking_container

# Load S&P 500 financial data
@st.cache_data
def load_sp500_data():
    """Load S&P 500 financial data from CSV"""
    try:
        df = pd.read_csv('sp500_financial_data.csv')
        # Filter only successful data entries
        df = df[df['status'] == 'success']
        return df
    except Exception as e:
        st.error(f"Error loading S&P 500 data: {str(e)}")
        return pd.DataFrame()

# Valuation calculation functions with detailed explanations
def calculate_asset_based_value(row, show_calculation=False):
    """Calculate asset-based valuation with detailed breakdown"""
    try:
        total_assets = row['assets'] if pd.notna(row['assets']) else 0
        total_liabilities = row['liabilities'] if pd.notna(row['liabilities']) else 0
        goodwill = row['goodwill'] if pd.notna(row['goodwill']) else 0
        intangible_assets = row['intangible_assets'] if pd.notna(row['intangible_assets']) else 0
        
        # Calculate book value
        book_value = total_assets - total_liabilities
        
        # Calculate tangible book value
        tangible_book_value = book_value - goodwill - intangible_assets
        
        # Apply conservative estimate
        conservative_estimate = book_value * 0.8
        fair_value = max(tangible_book_value, conservative_estimate)
        
        if show_calculation:
            calculation_details = {
                'total_assets': total_assets,
                'total_liabilities': total_liabilities,
                'goodwill': goodwill,
                'intangible_assets': intangible_assets,
                'book_value': book_value,
                'tangible_book_value': tangible_book_value,
                'conservative_estimate': conservative_estimate,
                'final_fair_value': fair_value
            }
            return fair_value, calculation_details
        
        return fair_value
    except:
        return 0

def calculate_dcf_value(row, risk_free_rate=0.045, market_risk_premium=0.06, show_calculation=False):
    """Calculate DCF valuation with detailed breakdown"""
    try:
        # Get free cash flow
        fcf = row['free_cash_flow'] if pd.notna(row['free_cash_flow']) else 0
        
        # If no FCF, try to calculate from operating cash flow
        if fcf == 0:
            operating_cf = row['net_cash_flow_from_operating_activities'] if pd.notna(row['net_cash_flow_from_operating_activities']) else 0
            capex = abs(row['capital_expenditure']) if pd.notna(row['capital_expenditure']) else 0
            fcf = operating_cf - capex
        
        if fcf <= 0:
            return 0
        
        # Estimate beta (simplified - in practice would use market data)
        beta = 1.0  # Default beta
        
        # Calculate WACC
        cost_of_equity = risk_free_rate + beta * market_risk_premium
        wacc = cost_of_equity  # Simplified - no debt component
        
        # Growth assumptions
        growth_rate = 0.05  # 5% growth for 5 years
        terminal_growth = 0.025  # 2.5% terminal growth
        
        # Project FCF for 5 years
        projected_fcf = []
        for year in range(1, 6):
            projected_fcf.append(fcf * (1 + growth_rate) ** year)
        
        # Calculate terminal value
        terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (wacc - terminal_growth)
        
        # Discount to present value
        total_pv = sum([fcf / (1 + wacc) ** (i + 1) for i, fcf in enumerate(projected_fcf)])
        terminal_pv = terminal_value / (1 + wacc) ** 5
        
        enterprise_value = total_pv + terminal_pv
        
        # Convert to equity value (simplified)
        equity_value = enterprise_value
        
        if show_calculation:
            calculation_details = {
                'base_fcf': fcf,
                'beta': beta,
                'risk_free_rate': risk_free_rate,
                'market_risk_premium': market_risk_premium,
                'cost_of_equity': cost_of_equity,
                'wacc': wacc,
                'growth_rate': growth_rate,
                'terminal_growth': terminal_growth,
                'projected_fcf': projected_fcf,
                'terminal_value': terminal_value,
                'total_pv': total_pv,
                'terminal_pv': terminal_pv,
                'enterprise_value': enterprise_value,
                'final_equity_value': equity_value
            }
            return equity_value, calculation_details
        
        return equity_value
    except:
        return 0

def calculate_comparable_value(row, show_calculation=False):
    """Calculate comparable company valuation with detailed breakdown"""
    try:
        # Get key metrics
        revenue = row['revenues'] if pd.notna(row['revenues']) else 0
        net_income = row['net_income_loss'] if pd.notna(row['net_income_loss']) else 0
        book_value = row['book_value'] if pd.notna(row['book_value']) else 0
        
        if revenue <= 0 or net_income <= 0:
            return 0
        
        # Industry average multiples (simplified - would use actual peer data)
        ev_revenue_multiple = 2.0
        pe_multiple = 15.0
        pb_multiple = 1.5
        
        # Calculate valuations using different multiples
        ev_revenue_value = revenue * ev_revenue_multiple
        pe_value = net_income * pe_multiple
        pb_value = book_value * pb_multiple
        
        # Average the valuations
        avg_value = (ev_revenue_value + pe_value + pb_value) / 3
        
        if show_calculation:
            calculation_details = {
                'revenue': revenue,
                'net_income': net_income,
                'book_value': book_value,
                'ev_revenue_multiple': ev_revenue_multiple,
                'pe_multiple': pe_multiple,
                'pb_multiple': pb_multiple,
                'ev_revenue_value': ev_revenue_value,
                'pe_value': pe_value,
                'pb_value': pb_value,
                'final_avg_value': avg_value
            }
            return avg_value, calculation_details
        
        return avg_value
    except:
        return 0

def calculate_rankings(df, valuation_method, risk_free_rate=0.045, market_risk_premium=0.06, show_progress=True):
    """Calculate rankings based on selected valuation method with progress tracking"""
    results = []
    
    if show_progress:
        # Create a more sophisticated progress display
        progress_container = st.container()
        with progress_container:
            st.markdown("**📊 Processing Companies:**")
            progress_bar = st.progress(0)
            status_text = st.empty()
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                companies_processed = st.metric("Companies Processed", "0")
            with metrics_col2:
                successful_calculations = st.metric("Successful Calculations", "0")
            with metrics_col3:
                current_company = st.metric("Current Company", "Starting...")
    
    total_companies = len(df)
    successful_count = 0
    
    for idx, (_, row) in enumerate(df.iterrows()):
        try:
            ticker = row['ticker']
            company_name = row['company_name']
            sector = row['sector']
            
            if show_progress:
                progress = (idx + 1) / total_companies
                progress_bar.progress(progress)
                status_text.text(f"🔍 Analyzing {ticker} - {company_name}")
                
                # Update metrics
                companies_processed.metric("Companies Processed", f"{idx + 1}/{total_companies}")
                current_company.metric("Current Company", f"{ticker}")
            
            if valuation_method == "Asset Based Value":
                fair_value = calculate_asset_based_value(row)
            elif valuation_method == "Discounted Cash Flow Value":
                fair_value = calculate_dcf_value(row, risk_free_rate, market_risk_premium)
            elif valuation_method == "Comparable Company Analysis":
                fair_value = calculate_comparable_value(row)
            else:
                continue
            
            if fair_value > 0:
                successful_count += 1
                results.append({
                    'ticker': ticker,
                    'company_name': company_name,
                    'sector': sector,
                    'fair_value': fair_value,
                    'revenue': row['revenues'] if pd.notna(row['revenues']) else 0,
                    'net_income': row['net_income_loss'] if pd.notna(row['net_income_loss']) else 0,
                    'book_value': row['book_value'] if pd.notna(row['book_value']) else 0
                })
                
                if show_progress:
                    successful_calculations.metric("Successful Calculations", f"{successful_count}")
                    
        except Exception as e:
            continue
    
    if show_progress:
        # Final status update
        progress_bar.progress(1.0)
        status_text.text("✅ Analysis complete! Sorting results...")
        companies_processed.metric("Companies Processed", f"{total_companies}/{total_companies}")
        successful_calculations.metric("Successful Calculations", f"{successful_count}")
        current_company.metric("Status", "Complete")
        
        time.sleep(1.5)
        progress_container.empty()
    
    # Sort by fair value (descending)
    results.sort(key=lambda x: x['fair_value'], reverse=True)
    
    # Add ranking
    for i, result in enumerate(results):
        result['rank'] = i + 1
    
    return results

def show_parameter_confirmation(valuation_method, risk_free_rate=None, market_risk_premium=None):
    """Show parameter confirmation before calculations"""
    
    st.subheader("🔍 Parameter Confirmation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.info(f"**Valuation Method:** {valuation_method}")
        
        if valuation_method == "Asset Based Value":
            st.markdown("""
            **Methodology:**
            - Calculates tangible book value (Total Assets - Total Liabilities - Intangible Assets - Goodwill)
            - Applies conservative 80% discount to book value
            - Uses the higher of tangible book value or conservative estimate
            """)
            
        elif valuation_method == "Discounted Cash Flow Value":
            st.markdown(f"""
            **Methodology:**
            - Projects free cash flows for 5 years with {risk_free_rate*100:.1f}% risk-free rate
            - Uses {market_risk_premium*100:.1f}% market risk premium for discount rate calculation
            - Applies 5% growth rate for first 5 years, 2.5% terminal growth
            - Discounts all cash flows to present value using WACC
            """)
            
        elif valuation_method == "Comparable Company Analysis":
            st.markdown("""
            **Methodology:**
            - Uses industry average multiples: EV/Revenue (2.0x), P/E (15.0x), P/B (1.5x)
            - Calculates value using each multiple
            - Averages the three approaches for final valuation
            """)
    
    with col2:
        st.metric("Companies to Analyze", "503")
        st.metric("Data Quality", "High")
        st.metric("Processing Time", "~30 seconds")
    
    # Confirmation button
    if st.button("🚀 Start Analysis", type="primary"):
        return True
    
    return False

def show_calculation_example(valuation_method, risk_free_rate=None, market_risk_premium=None):
    """Show a sample calculation for the selected method"""
    
    st.subheader("🧮 Sample Calculation")
    
    # Load sample data
    df = load_sp500_data()
    if df.empty:
        return
    
    # Use a well-known company for the example
    sample_companies = ['AAPL', 'MSFT', 'GOOGL']
    sample_row = None
    
    for ticker in sample_companies:
        sample_data = df[df['ticker'] == ticker]
        if not sample_data.empty:
            sample_row = sample_data.iloc[0]
            break
    
    if sample_row is None:
        sample_row = df.iloc[0]
    
    ticker = sample_row['ticker']
    company_name = sample_row['company_name']
    
    st.info(f"**Example calculation for {ticker} - {company_name}**")
    
    if valuation_method == "Asset Based Value":
        fair_value, details = calculate_asset_based_value(sample_row, show_calculation=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Input Values:**")
            st.write(f"- Total Assets: ${details['total_assets']/1e9:.2f}B")
            st.write(f"- Total Liabilities: ${details['total_liabilities']/1e9:.2f}B")
            st.write(f"- Goodwill: ${details['goodwill']/1e9:.2f}B")
            st.write(f"- Intangible Assets: ${details['intangible_assets']/1e9:.2f}B")
        
        with col2:
            st.markdown("**Calculations:**")
            st.write(f"- Book Value: ${details['book_value']/1e9:.2f}B")
            st.write(f"- Tangible Book Value: ${details['tangible_book_value']/1e9:.2f}B")
            st.write(f"- Conservative Estimate: ${details['conservative_estimate']/1e9:.2f}B")
            st.write(f"**Final Fair Value: ${details['final_fair_value']/1e9:.2f}B**")
    
    elif valuation_method == "Discounted Cash Flow Value":
        fair_value, details = calculate_dcf_value(sample_row, risk_free_rate, market_risk_premium, show_calculation=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Key Parameters:**")
            st.write(f"- Base FCF: ${details['base_fcf']/1e6:.0f}M")
            st.write(f"- Risk-free Rate: {details['risk_free_rate']*100:.1f}%")
            st.write(f"- Market Risk Premium: {details['market_risk_premium']*100:.1f}%")
            st.write(f"- WACC: {details['wacc']*100:.1f}%")
            st.write(f"- Growth Rate: {details['growth_rate']*100:.1f}%")
        
        with col2:
            st.markdown("**Projected Cash Flows:**")
            for i, fcf in enumerate(details['projected_fcf']):
                st.write(f"- Year {i+1}: ${fcf/1e6:.0f}M")
            st.write(f"- Terminal Value: ${details['terminal_value']/1e9:.2f}B")
            st.write(f"**Final Enterprise Value: ${details['final_equity_value']/1e9:.2f}B**")
    
    elif valuation_method == "Comparable Company Analysis":
        fair_value, details = calculate_comparable_value(sample_row, show_calculation=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Financial Metrics:**")
            st.write(f"- Revenue: ${details['revenue']/1e9:.2f}B")
            st.write(f"- Net Income: ${details['net_income']/1e9:.2f}B")
            st.write(f"- Book Value: ${details['book_value']/1e9:.2f}B")
        
        with col2:
            st.markdown("**Valuation Multiples:**")
            st.write(f"- EV/Revenue: ${details['ev_revenue_value']/1e9:.2f}B")
            st.write(f"- P/E: ${details['pe_value']/1e9:.2f}B")
            st.write(f"- P/B: ${details['pb_value']/1e9:.2f}B")
            st.write(f"**Average Fair Value: ${details['final_avg_value']/1e9:.2f}B**")

class FinancialAnalysisAgent:
    def __init__(self, openai_api_key: str, langsmith_api_key: str = None):
        # Configure LangSmith tracing if API key is provided
        callbacks = []
        if langsmith_api_key:
            # Set environment variables for LangSmith
            os.environ["LANGCHAIN_API_KEY"] = langsmith_api_key
            os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_PROJECT"] = "financial-analysis-agent"
            
            # Add LangSmith tracer callback with project name
            callbacks.append(LangChainTracer(project_name="financial-analysis-agent"))
            
            # Initialize LangSmith client
            self.langsmith_client = Client(api_key=langsmith_api_key)
        else:
            # Disable tracing
            os.environ["LANGCHAIN_TRACING_V2"] = "false"
            self.langsmith_client = None
        
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=openai_api_key,
            callbacks=callbacks
        )
        
    def analyze_rankings(self, rankings_data, valuation_method, risk_free_rate=None, market_risk_premium=None):
        """Generate AI analysis of the rankings"""
        
        # Prepare the data for analysis
        top_companies = rankings_data[:10]  # Top 10 companies
        
        analysis_prompt = f"""
        Analyze the top 10 companies ranked by {valuation_method}:
        
        Top Companies:
        """
        
        for company in top_companies:
            analysis_prompt += f"""
        {company['rank']}. {company['ticker']} - {company['company_name']} ({company['sector']})
           Fair Value: ${company['fair_value']/1e9:.2f}B
           Revenue: ${company['revenue']/1e9:.2f}B
           Net Income: ${company['net_income']/1e9:.2f}B
           Book Value: ${company['book_value']/1e9:.2f}B
        """
        
        if valuation_method == "Discounted Cash Flow Value":
            analysis_prompt += f"""
        
        Valuation Parameters:
        - Risk-free Rate: {risk_free_rate:.2%}
        - Market Risk Premium: {market_risk_premium:.2%}
        """
        
        analysis_prompt += """
        
        Please provide a comprehensive analysis including:
        1. Key insights about the top-ranked companies
        2. Sector distribution and trends
        3. Valuation methodology considerations
        4. Potential risks and limitations
        5. Investment implications
        6. Key assumptions made in the valuation
        """
        
        response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
        return response.content

# Streamlit UI
def main():
    st.set_page_config(
        page_title="FinBot",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 AI Agentic Financial Bot a.k.a. FinBot")
    st.markdown("*Powered by LangChain, LangGraph, LangSmith & OpenAI*")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Keys
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    langsmith_api_key = st.sidebar.text_input("LangSmith API Key (Optional)", type="password")
    
    # Enable/disable LangSmith tracing
    enable_tracing = st.sidebar.checkbox("Enable LangSmith Tracing", value=False, 
                                       help="Enable this to track analysis in LangSmith (requires API key)")
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()
    
    # Load data
    df = load_sp500_data()
    if df.empty:
        st.error("Failed to load S&P 500 financial data.")
        st.stop()
    
    # Initialize session state for conversation
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 'welcome'
    
    if 'valuation_method' not in st.session_state:
        st.session_state.valuation_method = None
    
    if 'risk_free_rate' not in st.session_state:
        st.session_state.risk_free_rate = None
    
    if 'market_risk_premium' not in st.session_state:
        st.session_state.market_risk_premium = None
    
    if 'rankings' not in st.session_state:
        st.session_state.rankings = None
    
    if 'show_confirmation' not in st.session_state:
        st.session_state.show_confirmation = False
    
    if 'ai_analysis' not in st.session_state:
        st.session_state.ai_analysis = None
    
    # Display chat interface
    st.header("💬 Chat with AI Financial Analyst")
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask me about company rankings..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Show thinking UI
        with st.chat_message("assistant"):
            # Determine the type of processing needed
            prompt_lower = prompt.lower()
            is_general = is_general_question(prompt_lower)
            
            if is_general:
                # For general questions, show AI thinking with stages
                thinking_container = st.container()
                with thinking_container:
                    # Stage 1: Understanding the question
                    with st.spinner("🧠 Understanding your question..."):
                        time.sleep(0.6)
                    
                    # Stage 2: Searching knowledge base
                    with st.spinner("🔍 Searching knowledge base..."):
                        time.sleep(0.4)
                    
                    # Stage 3: Generating response
                    with st.spinner("✍️ Generating comprehensive response..."):
                        time.sleep(0.3)
                        
                        # Process the message
                        response = process_user_input(prompt, df, openai_api_key, langsmith_api_key if enable_tracing else None)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                # For ranking requests, show processing stages
                thinking_container = st.container()
                with thinking_container:
                    # Stage 1: Analyzing request
                    with st.spinner("📊 Analyzing your request..."):
                        time.sleep(0.4)
                    
                    # Stage 2: Processing
                    with st.spinner("⚙️ Processing parameters..."):
                        time.sleep(0.3)
                        
                        # Process the message
                        response = process_user_input(prompt, df, openai_api_key, langsmith_api_key if enable_tracing else None)
                        st.markdown(response)
                        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Display current step guidance
    if st.session_state.current_step == 'welcome':
        st.info("💡 **You can ask me about:**\n• Company rankings: 'I want to see company rankings based on fair market value'\n• General questions: 'Who is the CEO of Apple?' or 'What is DCF valuation?'\n• Financial concepts: 'Explain P/E ratio' or 'What is market cap?'")
    elif st.session_state.current_step == 'select_valuation':
        st.info("💡 **Please select a valuation method:** Asset Based Value, Discounted Cash Flow Value, or Comparable Company Analysis")
    elif st.session_state.current_step == 'dcf_parameters':
        st.info("💡 **For DCF analysis, please provide:** Risk-free rate (e.g., 4.5%) and Market Risk Premium (e.g., 6%)")
    elif st.session_state.current_step == 'show_results':
        st.info("💡 **Analysis complete!** You can ask for more details, start a new analysis, or ask general questions about companies and finance.")
    
    # Show parameter confirmation if needed
    if st.session_state.show_confirmation:
        st.divider()
        show_calculation_example(
            st.session_state.valuation_method, 
            st.session_state.risk_free_rate, 
            st.session_state.market_risk_premium
        )
        
        if show_parameter_confirmation(
            st.session_state.valuation_method, 
            st.session_state.risk_free_rate, 
            st.session_state.market_risk_premium
        ):
            # Start calculations
            st.session_state.show_confirmation = False
            st.session_state.current_step = 'calculating'
            
            # Calculate rankings with progress and thinking UI
            with st.spinner("🔢 Calculating valuations for 503 companies..."):
                rankings = calculate_rankings(
                    df, 
                    st.session_state.valuation_method, 
                    st.session_state.risk_free_rate, 
                    st.session_state.market_risk_premium
                )
            
            st.session_state.rankings = rankings
            st.session_state.current_step = 'show_results'
            
            # Generate AI analysis with thinking UI
            with st.spinner("🤖 AI is analyzing the rankings and generating insights..."):
                agent = FinancialAnalysisAgent(openai_api_key, langsmith_api_key if enable_tracing else None)
                analysis = agent.analyze_rankings(
                    rankings, 
                    st.session_state.valuation_method, 
                    st.session_state.risk_free_rate, 
                    st.session_state.market_risk_premium
                )
                
                st.session_state.ai_analysis = analysis
    
    # Display results if available
    if st.session_state.rankings is not None and st.session_state.current_step == 'show_results':
        with st.container(key="rankings_results_container"):
            display_rankings_results(st.session_state.rankings, st.session_state.valuation_method)
        
        # Display AI analysis if available
        if st.session_state.ai_analysis is not None:
            st.subheader("🤖 AI Analysis")
            st.markdown(st.session_state.ai_analysis)

def process_user_input(prompt, df, openai_api_key, langsmith_api_key):
    """Process user input and generate appropriate response"""
    
    prompt_lower = prompt.lower()
    
    # Check if this is a general question (not about rankings)
    if is_general_question(prompt_lower):
        return answer_general_question(prompt, openai_api_key, langsmith_api_key)
    
    # Welcome step - user wants to see rankings
    if st.session_state.current_step == 'welcome':
        if any(keyword in prompt_lower for keyword in ['rank', 'ranking', 'fair market value', 'company']):
            st.session_state.current_step = 'select_valuation'
            return """
I'd be happy to help you rank companies based on fair market value! 

I can calculate rankings using three different valuation methods:

1. **Asset Based Value** - Based on tangible book value and asset quality
2. **Discounted Cash Flow Value** - Based on projected future cash flows (requires macroeconomic parameters)
3. **Comparable Company Analysis** - Based on industry multiples and peer comparisons

Which valuation method would you prefer to use?
"""
    
    # Select valuation method step
    elif st.session_state.current_step == 'select_valuation':
        if 'asset' in prompt_lower and 'based' in prompt_lower:
            st.session_state.valuation_method = "Asset Based Value"
            st.session_state.show_confirmation = True
            return """
Great choice! I'll use the Asset Based Value method.

Let me show you how this calculation works and confirm the parameters before we start analyzing all 503 companies.
"""
        
        elif 'dcf' in prompt_lower or 'discounted' in prompt_lower or 'cash flow' in prompt_lower:
            st.session_state.valuation_method = "Discounted Cash Flow Value"
            st.session_state.current_step = 'dcf_parameters'
            return """
Great choice! For Discounted Cash Flow analysis, I need some macroeconomic parameters to calculate the discount rate.

Please provide:
1. **Risk-free rate** (e.g., 4.5% for current 10-year Treasury)
2. **Market Risk Premium** (e.g., 6% for typical equity risk premium)

You can provide these as percentages (e.g., "4.5% and 6%") or as decimals (e.g., "0.045 and 0.06").
"""
        
        elif 'comparable' in prompt_lower or 'peer' in prompt_lower or 'multiple' in prompt_lower:
            st.session_state.valuation_method = "Comparable Company Analysis"
            st.session_state.show_confirmation = True
            return """
Excellent choice! I'll use the Comparable Company Analysis method.

Let me show you how this calculation works and confirm the parameters before we start analyzing all 503 companies.
"""
        
        else:
            return """
I didn't quite understand your choice. Please specify one of these valuation methods:

- **Asset Based Value** - For asset-based valuation
- **Discounted Cash Flow Value** - For DCF analysis (requires additional parameters)
- **Comparable Company Analysis** - For peer-based valuation
"""
    
    # DCF parameters step
    elif st.session_state.current_step == 'dcf_parameters':
        # Extract numbers from the prompt
        import re
        numbers = re.findall(r'\d+\.?\d*', prompt)
        
        if len(numbers) >= 2:
            try:
                # Convert to float and handle percentage signs
                if '%' in prompt:
                    risk_free_rate = float(numbers[0]) / 100
                    market_risk_premium = float(numbers[1]) / 100
                else:
                    risk_free_rate = float(numbers[0])
                    market_risk_premium = float(numbers[1])
                
                st.session_state.risk_free_rate = risk_free_rate
                st.session_state.market_risk_premium = market_risk_premium
                st.session_state.show_confirmation = True
                
                return f"""
Perfect! I have your parameters:
- Risk-free Rate: {risk_free_rate:.2%}
- Market Risk Premium: {market_risk_premium:.2%}

Let me show you how the DCF calculation works and confirm everything before we start analyzing all 503 companies.
"""
            except ValueError:
                return "I couldn't parse the numbers correctly. Please provide the risk-free rate and market risk premium as percentages (e.g., '4.5% and 6%') or decimals (e.g., '0.045 and 0.06')."
        else:
            return "I need two numbers: the risk-free rate and market risk premium. Please provide them as percentages (e.g., '4.5% and 6%') or decimals (e.g., '0.045 and 0.06')."
    
    # Show results step - handle follow-up questions
    elif st.session_state.current_step == 'show_results':
        if any(keyword in prompt_lower for keyword in ['new', 'start over', 'reset', 'another']):
            # Reset to welcome
            st.session_state.current_step = 'welcome'
            st.session_state.valuation_method = None
            st.session_state.risk_free_rate = None
            st.session_state.market_risk_premium = None
            st.session_state.rankings = None
            st.session_state.show_confirmation = False
            st.session_state.ai_analysis = None
            return "Sure! Let's start over. What would you like to analyze?"
        
        elif any(keyword in prompt_lower for keyword in ['assumption', 'method', 'how', 'calculate']):
            return explain_valuation_method(st.session_state.valuation_method)
        
        elif any(keyword in prompt_lower for keyword in ['sector', 'industry', 'top']):
            return analyze_sector_distribution(st.session_state.rankings)
        
        else:
            return "I can help you with:\n- Starting a new analysis\n- Explaining the valuation method and assumptions\n- Analyzing sector distribution\n- Or any other questions about the rankings!"
    
    # Default response
    return "I'm here to help you with financial analysis! You can:\n\n• **Ask for company rankings**: 'I want to see company rankings based on fair market value'\n• **Ask general questions**: 'Who is the CEO of Apple?' or 'What is DCF valuation?'\n• **Learn about finance**: 'Explain P/E ratio' or 'What is market cap?'\n\nWhat would you like to know?"

def is_general_question(prompt_lower):
    """Determine if the user is asking a general question rather than requesting rankings"""
    
    # First, check for ranking-specific keywords that should NOT be treated as general questions
    ranking_keywords = [
        'rank', 'ranking', 'rankings', 'ranked',
        'fair market value', 'market value',
        'asset based', 'asset-based',
        'comparable company', 'comparable analysis',
        'valuation method', 'valuation methods'
    ]
    
    # If the prompt contains ranking keywords, it's not a general question
    for keyword in ranking_keywords:
        if keyword in prompt_lower:
            return False
    
    # Keywords that indicate general questions (but not ranking requests)
    general_keywords = [
        'who is', 'what is', 'when', 'where', 'why', 'how',
        'ceo', 'founder', 'president', 'executive', 'director',
        'founded', 'established', 'created', 'started',
        'headquarters', 'location', 'address', 'city', 'country',
        'industry', 'sector', 'business', 'company', 'corporation',
        'revenue', 'profit', 'earnings', 'sales', 'market cap',
        'stock price', 'share price', 'trading', 'exchange',
        'dividend', 'payout', 'yield', 'ratio', 'multiple',
        'valuation', 'worth', 'value', 'price',
        'competitor', 'rival', 'peer', 'similar',
        'product', 'service', 'brand', 'technology',
        'financial', 'accounting', 'investment', 'trading',
        'explain', 'define', 'describe', 'tell me about',
        'difference between', 'compare', 'versus', 'vs',
        'example', 'instance', 'case study'
    ]
    
    # Check if any general keywords are present
    for keyword in general_keywords:
        if keyword in prompt_lower:
            return True
    
    # Check for specific question patterns
    question_patterns = [
        'who is the ceo of',
        'who founded',
        'what does',
        'how does',
        'when was',
        'where is',
        'why is',
        'explain',
        'define',
        'describe',
        'tell me about'
    ]
    
    for pattern in question_patterns:
        if pattern in prompt_lower:
            return True
    
    return False

def answer_general_question(question, openai_api_key, langsmith_api_key):
    """Answer general questions using the LLM"""
    
    try:
        # Initialize the AI agent
        agent = FinancialAnalysisAgent(openai_api_key, langsmith_api_key)
        
        # Create a context-aware prompt for financial and business questions
        system_prompt = """You are an expert financial analyst and business intelligence assistant. You have deep knowledge of:

1. **Company Information**: CEOs, founders, executives, headquarters, history, business models
2. **Financial Concepts**: Valuation methods, ratios, metrics, investment strategies
3. **Market Analysis**: Industry trends, competitive landscapes, market dynamics
4. **Investment Topics**: Stocks, bonds, ETFs, mutual funds, portfolio management
5. **Business Strategy**: Corporate governance, mergers & acquisitions, strategic initiatives

Provide accurate, informative, and helpful responses. When discussing companies, include relevant financial context when appropriate. Be conversational but professional. If you're not certain about specific current information (like exact numbers or recent changes), acknowledge this and provide the most recent information you have.

Format your responses clearly with bullet points, sections, and emojis when appropriate to make them easy to read."""
        
        # Create the full prompt
        full_prompt = f"""
{system_prompt}

User Question: {question}

Please provide a comprehensive and helpful response. If this is about a specific company, include relevant financial context when possible.
"""
        
        # Get response from LLM with thinking indicators
        response = agent.llm.invoke([HumanMessage(content=full_prompt)])
        
        return response.content
        
    except Exception as e:
        return f"I apologize, but I encountered an error while processing your question: {str(e)}. Please try rephrasing your question or ask about company rankings instead."

def display_rankings_results(rankings, valuation_method):
    """Display rankings in a nice format"""
    
    st.header("📈 Rankings Results")
    
    # Create DataFrame for display
    df_display = pd.DataFrame(rankings)
    df_display['Fair Value (B)'] = df_display['fair_value'] / 1e9
    df_display['Revenue (B)'] = df_display['revenue'] / 1e9
    df_display['Net Income (B)'] = df_display['net_income'] / 1e9
    
    # Display top 20 companies
    st.subheader(f"Top 20 Companies by {valuation_method}")
    
    display_cols = ['rank', 'ticker', 'company_name', 'sector', 'Fair Value (B)', 'Revenue (B)', 'Net Income (B)']
    st.dataframe(
        df_display[display_cols].head(20),
        column_config={
            "rank": st.column_config.NumberColumn("Rank", format="%d"),
            "ticker": st.column_config.TextColumn("Ticker"),
            "company_name": st.column_config.TextColumn("Company"),
            "sector": st.column_config.TextColumn("Sector"),
            "Fair Value (B)": st.column_config.NumberColumn("Fair Value ($B)", format="%.2f"),
            "Revenue (B)": st.column_config.NumberColumn("Revenue ($B)", format="%.2f"),
            "Net Income (B)": st.column_config.NumberColumn("Net Income ($B)", format="%.2f"),
        },
        hide_index=True
    )
    
    # Create visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Sector distribution
        sector_counts = df_display.head(20)['sector'].value_counts()
        fig = px.pie(
            values=sector_counts.values, 
            names=sector_counts.index, 
            title="Sector Distribution (Top 20)"
        )
        st.plotly_chart(fig, use_container_width=True, key="sector_distribution")
    
    with col2:
        # Fair value vs revenue scatter
        fig = px.scatter(
            df_display.head(20),
            x='Revenue (B)',
            y='Fair Value (B)',
            hover_data=['ticker', 'company_name'],
            title="Fair Value vs Revenue (Top 20)"
        )
        st.plotly_chart(fig, use_container_width=True, key="fair_value_scatter")

def explain_valuation_method(method):
    """Explain the valuation method and assumptions"""
    
    explanations = {
        "Asset Based Value": """
## 🏢 Asset Based Value Method

**How it works:**
- Calculates tangible book value (Total Assets - Total Liabilities - Intangible Assets - Goodwill)
- Uses conservative estimates to account for asset quality
- Applies an 80% discount to book value as a safety margin

**Key Assumptions:**
- Tangible assets are more reliable than intangible assets
- Conservative approach to account for potential asset write-downs
- Suitable for asset-heavy businesses (utilities, real estate, manufacturing)

**Limitations:**
- May undervalue companies with strong intangible assets (brands, patents)
- Doesn't account for future earnings potential
- Less suitable for high-growth technology companies
""",
        
        "Discounted Cash Flow Value": """
## 💰 Discounted Cash Flow (DCF) Method

**How it works:**
- Projects future free cash flows for 5 years
- Calculates terminal value using perpetual growth model
- Discounts all cash flows to present value using WACC

**Key Assumptions:**
- 5% annual growth rate for first 5 years
- 2.5% terminal growth rate
- Beta of 1.0 (market average)
- WACC = Risk-free rate + Market risk premium

**Growth Projections:**
- Year 1-5: 5% annual growth
- Terminal: 2.5% perpetual growth
- Conservative estimates to account for uncertainty

**Limitations:**
- Highly sensitive to growth and discount rate assumptions
- Requires reliable cash flow projections
- May not capture cyclical or seasonal variations
""",
        
        "Comparable Company Analysis": """
## 🔍 Comparable Company Analysis

**How it works:**
- Uses industry average multiples to value companies
- Combines EV/Revenue, P/E, and P/B ratios
- Averages multiple approaches for balanced valuation

**Key Assumptions:**
- EV/Revenue multiple: 2.0x
- P/E multiple: 15.0x
- P/B multiple: 1.5x
- Industry averages based on conservative estimates

**Valuation Formula:**
Fair Value = (EV/Revenue Value + P/E Value + P/B Value) / 3

**Limitations:**
- Assumes companies are comparable within sectors
- Uses simplified industry averages
- May not account for company-specific factors
- Market multiples can be volatile
"""
    }
    
    return explanations.get(method, "Valuation method explanation not available.")

def analyze_sector_distribution(rankings):
    """Analyze sector distribution of rankings"""
    
    if not rankings:
        return "No rankings data available for sector analysis."
    
    df_analysis = pd.DataFrame(rankings)
    
    # Sector analysis
    sector_counts = df_analysis['sector'].value_counts()
    sector_avg_values = df_analysis.groupby('sector')['fair_value'].mean().sort_values(ascending=False)
    
    response = """
## 📊 Sector Analysis

**Sector Distribution:**
"""
    
    for sector, count in sector_counts.head(10).items():
        response += f"- {sector}: {count} companies\n"
    
    response += f"""

**Average Fair Value by Sector (Top 10):**
"""
    
    for sector, avg_value in sector_avg_values.head(10).items():
        response += f"- {sector}: ${avg_value/1e9:.2f}B\n"
    
    response += """

**Key Insights:**
- Sectors with higher representation may indicate better financial health
- Higher average fair values suggest sectors with strong fundamentals
- Consider sector-specific risks and growth prospects
"""
    
    return response

if __name__ == "__main__":
    main()