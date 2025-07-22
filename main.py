import streamlit as st
import yfinance as yf
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
# from langgraph.prebuilt import ToolExecutor
# from langgraph.prebuilt import ToolNode

from langchain.tools import BaseTool
from langchain.callbacks.tracers import LangChainTracer
import os
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Initialize LangSmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "financial-analysis-agent"
import os
os.environ["LANGCHAIN_API_KEY"] = "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"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"



from typing import TypedDict
from typing import Annotated

class AnalysisState(TypedDict, total=False):
    ticker: Annotated[str, "multiple"]
    duration: str
    risk_free_rate: float
    market_risk_premium: float
    financial_data: dict
    comparable_data: dict
    market_data: dict
    period: str
    market_price_analysis: dict
    comparable_analysis: dict
    dcf_analysis: dict
    asset_analysis: dict
    final_synthesis: dict





@dataclass
class AnalysisParameters:
    ticker: str
    duration: str
    analysis_depth: str = "comprehensive"
    include_comparables: bool = True
    risk_free_rate: float = 0.045  # Current 10Y Treasury
    market_risk_premium: float = 0.06
    growth_rate: float = 0.025

class FinancialDataTool(BaseTool):
    name: str = "financial_data_extractor"
    description: str = "Extracts comprehensive financial data for a given ticker"
    
    def _run(self, ticker: str, period: str = "5y") -> Dict[str, Any]:
        try:
            stock = yf.Ticker(ticker)
            
            # Get historical data
            hist_data = stock.history(period=period)
            
            # Get financial statements
            financials = stock.financials
            balance_sheet = stock.balance_sheet
            cash_flow = stock.cashflow
            
            # Get stock info
            info = stock.info
            
            # Get analyst recommendations
            recommendations = stock.recommendations
            
            return {
                "historical_data": hist_data,
                "financials": financials,
                "balance_sheet": balance_sheet,
                "cash_flow": cash_flow,
                "info": info,
                "recommendations": recommendations,
                "success": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class ComparableCompanyTool(BaseTool):
    name: str = "comparable_company_finder"
    description: str = "Finds comparable companies in the same industry"
    
    def _run(self, ticker: str, industry: str, market_cap: float) -> Dict[str, Any]:
        try:
            # This would ideally use a more sophisticated API like Bloomberg or FactSet
            # For demo purposes, we'll use a simplified approach
            stock = yf.Ticker(ticker)
            sector = stock.info.get('sector', '')
            
            # Get some comparable tickers (this is simplified - in production, use proper screeners)
            comparable_tickers = self._get_comparable_tickers(sector)
            
            comparables_data = {}
            for comp_ticker in comparable_tickers[:5]:  # Limit to 5 comparables
                try:
                    comp_stock = yf.Ticker(comp_ticker)
                    comp_info = comp_stock.info
                    if comp_info.get('marketCap', 0) > 0:
                        comparables_data[comp_ticker] = comp_info
                except:
                    continue
            
            return {
                "comparables": comparables_data,
                "success": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _get_comparable_tickers(self, sector: str) -> List[str]:
        # Simplified mapping - in production, use proper industry classification
        sector_mapping = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'CRM', 'ADBE', 'ORCL'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'MRK', 'CVS', 'ABBV', 'TMO'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'GS', 'MS', 'C', 'BLK'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'NKE', 'MCD', 'SBUX'],
            'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG', 'PXD'],
            'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'UNP', 'LMT'],
        }
        return sector_mapping.get(sector, ['SPY', 'QQQ', 'DIA'])  # Default to ETFs

class MarketDataTool(BaseTool):
    name: str = "market_data_collector"
    description: str = "Collects broader market data and economic indicators"
    
    def _run(self, period: str = "5y") -> Dict[str, Any]:
        try:
            # Get market indices
            sp500 = yf.Ticker("^GSPC").history(period=period)
            nasdaq = yf.Ticker("^IXIC").history(period=period)
            dow = yf.Ticker("^DJI").history(period=period)
            
            # Get treasury rates (using ETFs as proxy)
            ten_year = yf.Ticker("^TNX").history(period=period)
            
            # Get VIX for volatility
            vix = yf.Ticker("^VIX").history(period=period)
            
            return {
                "sp500": sp500,
                "nasdaq": nasdaq,
                "dow": dow,
                "ten_year_treasury": ten_year,
                "vix": vix,
                "success": True
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

class FinancialAnalysisAgent:
    def __init__(self, openai_api_key: str, langsmith_api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4o",
            temperature=0.1,
            openai_api_key=openai_api_key
        )
        
        # Initialize tools
        self.financial_tool = FinancialDataTool()
        self.comparable_tool = ComparableCompanyTool()
        self.market_tool = MarketDataTool()
        
        # Initialize LangSmith client
        if langsmith_api_key:
            self.langsmith_client = Client(api_key=langsmith_api_key)
        
    def create_analysis_graph(self):
        """Create the LangGraph workflow for financial analysis"""
        
        def data_collection_node(state):
            """Node 1: Collect all financial data"""
            ticker = state['ticker']
            duration = state['duration']
            
            # Convert duration to yfinance period
            period_mapping = {
                "1 Month": "1mo",
                "3 Months": "3mo", 
                "6 Months": "6mo",
                "1 Year": "1y",
                "2 Years": "2y",
                "5 Years": "5y"
            }
            period = period_mapping.get(duration, "1y")
            
            # Collect financial data
            financial_data = self.financial_tool._run(ticker, period)
            comparable_data = self.comparable_tool._run(
                ticker, 
                financial_data['info'].get('sector', ''), 
                financial_data['info'].get('marketCap', 0)
            )
            market_data = self.market_tool._run(period)
            
            state['financial_data'] = financial_data
            state['comparable_data'] = comparable_data
            state['market_data'] = market_data
            state['period'] = period
            
            return state
        
        def market_price_analysis_node(state):
            """Node 2: Market Price Method Analysis"""
            financial_data = state['financial_data']
            
            if not financial_data['success']:
                return {'market_price_analysis': {"error": "Failed to get financial data"}}
            
            info = financial_data['info']
            hist_data = financial_data['historical_data']
            
            # Calculate key metrics
            current_price = hist_data['Close'].iloc[-1]
            market_cap = info.get('marketCap', 0)
            book_value = info.get('bookValue', 0)
            pe_ratio = info.get('trailingPE', 0)
            pb_ratio = info.get('priceToBook', 0)
            
            # Price volatility analysis
            returns = hist_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Support and resistance levels
            high_52w = hist_data['High'].rolling(window=252).max().iloc[-1]
            low_52w = hist_data['Low'].rolling(window=252).min().iloc[-1]
            
            analysis_prompt = f"""
            Analyze the market price method for {state['ticker']}:
            
            Current Price: ${current_price:.2f}
            Market Cap: ${market_cap:,.0f}
            Book Value per Share: ${book_value:.2f}
            P/E Ratio: {pe_ratio:.2f}
            P/B Ratio: {pb_ratio:.2f}
            Annualized Volatility: {volatility:.2%}
            52-Week High: ${high_52w:.2f}
            52-Week Low: ${low_52w:.2f}
            
            Provide detailed analysis using Chain of Thought reasoning:
            1. Market sentiment analysis
            2. Valuation ratios interpretation
            3. Price momentum and technical indicators
            4. Fair value estimate based on market metrics
            """
            
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            
            return {'market_price_analysis': {
                "analysis": response.content,
                "metrics": {
                    "current_price": current_price,
                    "market_cap": market_cap,
                    "pe_ratio": pe_ratio,
                    "pb_ratio": pb_ratio,
                    "volatility": volatility,
                    "high_52w": high_52w,
                    "low_52w": low_52w
                }
            }}
        
        def comparable_analysis_node(state):
            """Node 3: Comparable Company Analysis"""
            comparable_data = state['comparable_data']
            financial_data = state['financial_data']
            
            if not comparable_data['success']:
                return {'comparable_analysis': {"error": "Failed to get comparable data"}}
            
            target_info = financial_data['info']
            comparables = comparable_data['comparables']
            
            # Calculate relative metrics
            comp_metrics = []
            for ticker, info in comparables.items():
                comp_metrics.append({
                    'ticker': ticker,
                    'pe_ratio': info.get('trailingPE', 0),
                    'pb_ratio': info.get('priceToBook', 0),
                    'ev_revenue': info.get('enterpriseToRevenue', 0),
                    'market_cap': info.get('marketCap', 0),
                    'roe': info.get('returnOnEquity', 0)
                })
            
            # Calculate averages
            if comp_metrics:
                avg_pe = np.mean([m['pe_ratio'] for m in comp_metrics if m['pe_ratio'] > 0])
                avg_pb = np.mean([m['pb_ratio'] for m in comp_metrics if m['pb_ratio'] > 0])
                avg_ev_revenue = np.mean([m['ev_revenue'] for m in comp_metrics if m['ev_revenue'] > 0])
            else:
                avg_pe = avg_pb = avg_ev_revenue = 0
            
            analysis_prompt = f"""
            Perform comparable company analysis for {state['ticker']}:
            
            Target Company Metrics:
            - P/E Ratio: {target_info.get('trailingPE', 0):.2f}
            - P/B Ratio: {target_info.get('priceToBook', 0):.2f}
            - Enterprise Value/Revenue: {target_info.get('enterpriseToRevenue', 0):.2f}
            
            Comparable Companies Average:
            - Average P/E: {avg_pe:.2f}
            - Average P/B: {avg_pb:.2f}
            - Average EV/Revenue: {avg_ev_revenue:.2f}
            
            Comparable Companies: {list(comparables.keys())}
            
            Using Chain of Thought reasoning:
            1. Analyze relative valuation position
            2. Identify premium/discount to peers
            3. Consider qualitative factors (growth, profitability, risk)
            4. Calculate fair value estimate based on peer multiples
            """
            
            response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
            
            return {'comparable_analysis': {
                "analysis": response.content,
                "target_metrics": {
                    "pe_ratio": target_info.get('trailingPE', 0),
                    "pb_ratio": target_info.get('priceToBook', 0),
                    "ev_revenue": target_info.get('enterpriseToRevenue', 0)
                },
                "peer_averages": {
                    "avg_pe": avg_pe,
                    "avg_pb": avg_pb,
                    "avg_ev_revenue": avg_ev_revenue
                },
                "comparables": comp_metrics
            }}
        
        def dcf_analysis_node(state):
            """Node 4: Discounted Cash Flow Analysis"""
            financial_data = state['financial_data']
            
            if not financial_data['success']:
                return {'dcf_analysis': {"error": "Failed to get financial data"}}
            
            try:
                cash_flow = financial_data['cash_flow']
                info = financial_data['info']
                
                # Extract key DCF inputs
                if not cash_flow.empty and len(cash_flow.columns) > 0:
                    recent_fcf = cash_flow.loc['Free Cash Flow'].iloc[0] if 'Free Cash Flow' in cash_flow.index else 0
                    if recent_fcf == 0:
                        recent_fcf = cash_flow.loc['Total Cash From Operating Activities'].iloc[0] if 'Total Cash From Operating Activities' in cash_flow.index else 0
                else:
                    recent_fcf = info.get('freeCashflow', 0)
                
                shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 1))
                beta = info.get('beta', 1.0)
                
                # WACC calculation
                risk_free_rate = 0.045  # 10Y Treasury
                market_risk_premium = 0.06
                cost_of_equity = risk_free_rate + beta * market_risk_premium
                
                analysis_prompt = f"""
                Perform DCF analysis for {state['ticker']}:
                
                Key Inputs:
                - Most Recent Free Cash Flow: ${recent_fcf:,.0f}
                - Shares Outstanding: {shares_outstanding:,.0f}
                - Beta: {beta:.2f}
                - Cost of Equity (WACC): {cost_of_equity:.2%}
                - Risk-free Rate: {risk_free_rate:.2%}
                - Market Risk Premium: {market_risk_premium:.2%}
                
                Using Chain of Thought reasoning:
                1. Project future cash flows (5-year forecast)
                2. Calculate terminal value
                3. Discount to present value
                4. Calculate per-share intrinsic value
                5. Perform sensitivity analysis on growth and discount rates
                """
                
                response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
                
                # Simple DCF calculation for demonstration
                growth_rate = 0.05  # 5% growth assumption
                terminal_growth = 0.025  # 2.5% terminal growth
                
                # Project 5 years of FCF
                projected_fcf = []
                for year in range(1, 6):
                    projected_fcf.append(recent_fcf * (1 + growth_rate) ** year)
                
                # Terminal value
                terminal_value = projected_fcf[-1] * (1 + terminal_growth) / (cost_of_equity - terminal_growth)
                
                # Discount to present value
                total_pv = sum([fcf / (1 + cost_of_equity) ** (i + 1) for i, fcf in enumerate(projected_fcf)])
                terminal_pv = terminal_value / (1 + cost_of_equity) ** 5
                
                enterprise_value = total_pv + terminal_pv
                equity_value = enterprise_value  # Simplified - should subtract net debt
                fair_value_per_share = equity_value / shares_outstanding
                
                return {'dcf_analysis': {
                    "analysis": response.content,
                    "calculations": {
                        "recent_fcf": recent_fcf,
                        "wacc": cost_of_equity,
                        "projected_fcf": projected_fcf,
                        "terminal_value": terminal_value,
                        "enterprise_value": enterprise_value,
                        "fair_value_per_share": fair_value_per_share
                    }
                }}
                
            except Exception as e:
                return {'dcf_analysis': {"error": f"DCF calculation error: {str(e)}"}}
        
        def asset_based_analysis_node(state):
            """Node 5: Asset-Based Valuation"""
            financial_data = state['financial_data']
            
            if not financial_data['success']:
                return {'asset_analysis': {"error": "Failed to get financial data"}}
            
            try:
                balance_sheet = financial_data['balance_sheet']
                info = financial_data['info']
                
                if not balance_sheet.empty and len(balance_sheet.columns) > 0:
                    total_assets = balance_sheet.loc['Total Assets'].iloc[0] if 'Total Assets' in balance_sheet.index else 0
                    total_liabilities = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0] if 'Total Liabilities Net Minority Interest' in balance_sheet.index else 0
                    book_value = total_assets - total_liabilities
                else:
                    book_value = info.get('totalStockholderEquity', 0)
                    total_assets = info.get('totalAssets', 0)
                    total_liabilities = total_assets - book_value
                
                shares_outstanding = info.get('sharesOutstanding', info.get('impliedSharesOutstanding', 1))
                book_value_per_share = book_value / shares_outstanding
                
                analysis_prompt = f"""
                Perform asset-based valuation for {state['ticker']}:
                
                Balance Sheet Analysis:
                - Total Assets: ${total_assets:,.0f}
                - Total Liabilities: ${total_liabilities:,.0f}
                - Book Value of Equity: ${book_value:,.0f}
                - Book Value per Share: ${book_value_per_share:.2f}
                - Shares Outstanding: {shares_outstanding:,.0f}
                
                Using Chain of Thought reasoning:
                1. Analyze asset quality and composition
                2. Consider asset revaluation potential
                3. Evaluate liquidation vs going-concern value
                4. Compare to market value for value assessment
                """
                
                response = self.llm.invoke([HumanMessage(content=analysis_prompt)])
                
                return {'asset_analysis': {
                    "analysis": response.content,
                    "metrics": {
                        "total_assets": total_assets,
                        "total_liabilities": total_liabilities,
                        "book_value": book_value,
                        "book_value_per_share": book_value_per_share,
                        "shares_outstanding": shares_outstanding
                    }
                }}
                
            except Exception as e:
                return {'asset_analysis': {"error": f"Asset analysis error: {str(e)}"}}
        
        def synthesis_node(state):
            """Node 6: Synthesize all analyses into final recommendation"""
            
            # Collect all analysis results
            market_analysis = state.get('market_price_analysis', {})
            comparable_analysis = state.get('comparable_analysis', {})
            dcf_analysis = state.get('dcf_analysis', {})
            asset_analysis = state.get('asset_analysis', {})
            
            synthesis_prompt = f"""
            Synthesize comprehensive fair value analysis for {state['ticker']}:
            
            MARKET PRICE ANALYSIS:
            {market_analysis.get('analysis', 'Not available')}
            
            COMPARABLE COMPANY ANALYSIS:
            {comparable_analysis.get('analysis', 'Not available')}
            
            DISCOUNTED CASH FLOW ANALYSIS:
            {dcf_analysis.get('analysis', 'Not available')}
            
            ASSET-BASED VALUATION:
            {asset_analysis.get('analysis', 'Not available')}
            
            Using Chain of Thought reasoning:
            1. Weight each valuation method appropriately
            2. Consider the reliability and relevance of each approach
            3. Identify key risks and uncertainties
            4. Provide final fair value range and recommendation
            5. Suggest investment thesis (Buy/Hold/Sell with rationale)
            """
            
            response = self.llm.invoke([HumanMessage(content=synthesis_prompt)])
            
            return {'final_synthesis': {
                "comprehensive_analysis": response.content,
                "timestamp": datetime.now().isoformat()
            }}
        

        # Create the graph
        graph = StateGraph(AnalysisState)
        
        # Add nodes
        graph.add_node("data_collection", data_collection_node)
        graph.add_node("market_price_analysis", market_price_analysis_node)
        graph.add_node("comparable_analysis", comparable_analysis_node)
        graph.add_node("dcf_analysis", dcf_analysis_node)
        graph.add_node("asset_analysis", asset_based_analysis_node)
        graph.add_node("synthesis", synthesis_node)
        
        # Add edges
        graph.add_edge("data_collection", "market_price_analysis")
        graph.add_edge("data_collection", "comparable_analysis")
        graph.add_edge("data_collection", "dcf_analysis")
        graph.add_edge("data_collection", "asset_analysis")
        
        graph.add_edge("market_price_analysis", "synthesis")
        graph.add_edge("comparable_analysis", "synthesis")
        graph.add_edge("dcf_analysis", "synthesis")
        graph.add_edge("asset_analysis", "synthesis")
        
        graph.add_edge("synthesis", END)
        
        # Set entry point
        graph.set_entry_point("data_collection")
        
        return graph.compile()

# Streamlit UI
def main():
    st.set_page_config(
        page_title="AI Agentic Financial Analysis System",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🤖 AI Agentic Financial Analysis System")
    st.markdown("*Powered by LangChain, LangGraph, LangSmith & OpenAI*")
    
    # Sidebar for configuration
    st.sidebar.header("🔧 Configuration")
    
    # API Keys
    openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
    langsmith_api_key = st.sidebar.text_input("LangSmith API Key (Optional)", type="password")
    
    if not openai_api_key:
        st.warning("Please enter your OpenAI API key to proceed.")
        st.stop()
    
    # Analysis Parameters
    st.sidebar.header("📊 Analysis Parameters")
    ticker = st.sidebar.text_input("Stock Ticker", value="AAPL", help="Enter US stock ticker (e.g., AAPL, MSFT, GOOGL)")
    
    duration = st.sidebar.selectbox(
        "Analysis Duration",
        ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"],
        index=3
    )
    
    analysis_depth = st.sidebar.selectbox(
        "Analysis Depth",
        ["Quick Overview", "Standard Analysis", "Comprehensive Deep-Dive"],
        index=2
    )
    
    # Advanced parameters
    st.sidebar.subheader("Advanced Parameters")
    risk_free_rate = st.sidebar.slider("Risk-free Rate (%)", 0.0, 10.0, 4.5, 0.1) / 100
    market_risk_premium = st.sidebar.slider("Market Risk Premium (%)", 0.0, 15.0, 6.0, 0.1) / 100
    
    # Analysis button
    if st.sidebar.button("🚀 Start Analysis", type="primary"):
        if ticker:
            # Initialize the agent
            with st.spinner("Initializing AI Financial Analysis Agent..."):
                agent = FinancialAnalysisAgent(openai_api_key, langsmith_api_key)
                analysis_graph = agent.create_analysis_graph()
            
            # Create analysis parameters
            params = AnalysisParameters(
                ticker=ticker.upper(),
                duration=duration,
                analysis_depth=analysis_depth.lower(),
                risk_free_rate=risk_free_rate,
                market_risk_premium=market_risk_premium
            )
            
            # Initialize state
            initial_state = {
                "ticker": params.ticker,
                "duration": params.duration,
                "risk_free_rate": params.risk_free_rate,
                "market_risk_premium": params.market_risk_premium
            }
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Run the analysis
            try:
                status_text.text("🔍 Collecting financial data...")
                progress_bar.progress(20)
                
                # Execute the graph
                result = analysis_graph.invoke(initial_state)
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis Complete!")
                
                # Display results
                display_analysis_results(result, params)
                
            except Exception as e:
                st.error(f"Analysis failed: {str(e)}")
                st.exception(e)
        else:
            st.sidebar.error("Please enter a stock ticker.")

def display_analysis_results(result: Dict[str, Any], params: AnalysisParameters):
    """Display comprehensive analysis results"""
    
    ticker = params.ticker
    
    # Header with stock info
    if 'financial_data' in result and result['financial_data']['success']:
        info = result['financial_data']['info']
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Company", info.get('shortName', ticker))
        with col2:
            current_price = info.get('currentPrice', 0)
            st.metric("Current Price", f"${current_price:.2f}")
        with col3:
            market_cap = info.get('marketCap', 0)
            st.metric("Market Cap", f"${market_cap/1e9:.2f}B")
        with col4:
            pe_ratio = info.get('trailingPE', 0)
            st.metric("P/E Ratio", f"{pe_ratio:.2f}")
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "💰 Market Price", "🔍 Comparables", 
        "💸 DCF Analysis", "🏢 Asset-Based", "🎯 Final Synthesis"
    ])
    
    with tab1:
        st.header("📊 Analysis Overview")
        display_overview_charts(result, params)
    
    with tab2:
        st.header("💰 Market Price Method Analysis")
        if 'market_price_analysis' in result:
            display_market_price_analysis(result['market_price_analysis'], result)
    
    with tab3:
        st.header("🔍 Comparable Company Analysis")
        if 'comparable_analysis' in result:
            display_comparable_analysis(result['comparable_analysis'])
    
    with tab4:
        st.header("💸 Discounted Cash Flow Analysis")
        if 'dcf_analysis' in result:
            display_dcf_analysis(result['dcf_analysis'])
    
    with tab5:
        st.header("🏢 Asset-Based Valuation")
        if 'asset_analysis' in result:
            display_asset_analysis(result['asset_analysis'])
    
    with tab6:
        st.header("🎯 Final Synthesis & Recommendation")
        if 'final_synthesis' in result:
            display_final_synthesis(result['final_synthesis'])

def display_overview_charts(result: Dict[str, Any], params: AnalysisParameters):
    """Display overview charts and key metrics"""
    
    if 'financial_data' not in result or not result['financial_data']['success']:
        st.error("Failed to load financial data for overview charts.")
        return
    
    hist_data = result['financial_data']['historical_data']
    info = result['financial_data']['info']
    
    # Price chart with volume
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Stock Price', 'Volume'),
        row_width=[0.7, 0.3]
    )
    
    # Price chart
    fig.add_trace(
        go.Candlestick(
            x=hist_data.index,
            open=hist_data['Open'],
            high=hist_data['High'],
            low=hist_data['Low'],
            close=hist_data['Close'],
            name='Price'
        ),
        row=1, col=1
    )
    
    # Volume chart
    fig.add_trace(
        go.Bar(
            x=hist_data.index,
            y=hist_data['Volume'],
            name='Volume',
            marker_color='rgba(158,202,225,0.8)'
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title=f"{params.ticker} - Stock Price and Volume ({params.duration})",
        xaxis_rangeslider_visible=False,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Key financial metrics summary
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Performance Metrics")
        
        # Calculate returns
        returns = hist_data['Close'].pct_change().dropna()
        total_return = (hist_data['Close'].iloc[-1] / hist_data['Close'].iloc[0] - 1) * 100
        volatility = returns.std() * np.sqrt(252) * 100
        
        metrics_data = {
            'Metric': ['Total Return', 'Annualized Volatility', 'Sharpe Ratio', 'Max Drawdown'],
            'Value': [
                f"{total_return:.2f}%",
                f"{volatility:.2f}%",
                f"{(returns.mean() / returns.std() * np.sqrt(252)):.2f}",
                f"{((hist_data['Close'] / hist_data['Close'].cummax() - 1).min() * 100):.2f}%"
            ]
        }
        
        st.dataframe(pd.DataFrame(metrics_data), hide_index=True)
    
    with col2:
        st.subheader("🏢 Company Fundamentals")
        
        fundamentals_data = {
            'Metric': ['Sector', 'Industry', 'Employees', 'Beta', 'Forward P/E'],
            'Value': [
                info.get('sector', 'N/A'),
                info.get('industry', 'N/A')[:30] + '...' if len(info.get('industry', '')) > 30 else info.get('industry', 'N/A'),
                f"{info.get('fullTimeEmployees', 0):,}",
                f"{info.get('beta', 0):.2f}",
                f"{info.get('forwardPE', 0):.2f}"
            ]
        }
        
        st.dataframe(pd.DataFrame(fundamentals_data), hide_index=True)

def display_market_price_analysis(analysis: Dict[str, Any], result: Dict[str, Any]):
    """Display market price method analysis"""
    
    if 'error' in analysis:
        st.error(f"Market Price Analysis Error: {analysis['error']}")
        return
    
    # Display the AI analysis
    st.subheader("🤖 AI Analysis (Chain of Thought)")
    st.markdown(analysis['analysis'])
    
    # Display key metrics
    if 'metrics' in analysis:
        metrics = analysis['metrics']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Current Price", f"${metrics['current_price']:.2f}")
            st.metric("P/E Ratio", f"{metrics['pe_ratio']:.2f}")
        
        with col2:
            st.metric("P/B Ratio", f"{metrics['pb_ratio']:.2f}")
            st.metric("Volatility", f"{metrics['volatility']:.2%}")
        
        with col3:
            st.metric("52W High", f"${metrics['high_52w']:.2f}")
            st.metric("52W Low", f"${metrics['low_52w']:.2f}")
        
        # Price range visualization
        fig = go.Figure()
        
        current_price = metrics['current_price']
        high_52w = metrics['high_52w']
        low_52w = metrics['low_52w']
        
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=current_price,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Price Position in 52W Range"},
            gauge={
                'axis': {'range': [low_52w * 0.9, high_52w * 1.1]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [low_52w * 0.9, low_52w], 'color': "lightgray"},
                    {'range': [high_52w, high_52w * 1.1], 'color': "lightgray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': current_price
                }
            }
        ))
        
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def display_comparable_analysis(analysis: Dict[str, Any]):
    """Display comparable company analysis"""
    
    if 'error' in analysis:
        st.error(f"Comparable Analysis Error: {analysis['error']}")
        return
    
    # Display the AI analysis
    st.subheader("🤖 AI Analysis (Chain of Thought)")
    st.markdown(analysis['analysis'])
    
    if 'target_metrics' in analysis and 'peer_averages' in analysis:
        # Comparison metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Target Company")
            target = analysis['target_metrics']
            st.metric("P/E Ratio", f"{target['pe_ratio']:.2f}")
            st.metric("P/B Ratio", f"{target['pb_ratio']:.2f}")
            st.metric("EV/Revenue", f"{target['ev_revenue']:.2f}")
        
        with col2:
            st.subheader("👥 Peer Average")
            peers = analysis['peer_averages']
            st.metric("P/E Ratio", f"{peers['avg_pe']:.2f}")
            st.metric("P/B Ratio", f"{peers['avg_pb']:.2f}")
            st.metric("EV/Revenue", f"{peers['avg_ev_revenue']:.2f}")
        
        # Comparative visualization
        if 'comparables' in analysis:
            comp_df = pd.DataFrame(analysis['comparables'])
            
            if not comp_df.empty:
                fig = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('P/E Ratio', 'P/B Ratio', 'EV/Revenue')
                )
                
                # P/E comparison
                fig.add_trace(
                    go.Bar(x=comp_df['ticker'], y=comp_df['pe_ratio'], name='P/E'),
                    row=1, col=1
                )
                
                # P/B comparison
                fig.add_trace(
                    go.Bar(x=comp_df['ticker'], y=comp_df['pb_ratio'], name='P/B'),
                    row=1, col=2
                )
                
                # EV/Revenue comparison
                fig.add_trace(
                    go.Bar(x=comp_df['ticker'], y=comp_df['ev_revenue'], name='EV/Rev'),
                    row=1, col=3
                )
                
                fig.update_layout(
                    title="Valuation Multiples Comparison",
                    showlegend=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)

def display_dcf_analysis(analysis: Dict[str, Any]):
    """Display DCF analysis"""
    
    if 'error' in analysis:
        st.error(f"DCF Analysis Error: {analysis['error']}")
        return
    
    # Display the AI analysis
    st.subheader("🤖 AI Analysis (Chain of Thought)")
    st.markdown(analysis['analysis'])
    
    if 'calculations' in analysis:
        calc = analysis['calculations']
        
        # Key DCF metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Recent FCF", f"${calc['recent_fcf']/1e6:.0f}M")
            st.metric("WACC", f"{calc['wacc']:.2%}")
        
        with col2:
            st.metric("Enterprise Value", f"${calc['enterprise_value']/1e9:.2f}B")
            st.metric("Terminal Value", f"${calc['terminal_value']/1e9:.2f}B")
        
        with col3:
            st.metric("Fair Value/Share", f"${calc['fair_value_per_share']:.2f}", 
                     help="Intrinsic value based on DCF model")
        
        # Cash flow projection chart
        years = list(range(1, 6))
        projected_fcf = [fcf/1e6 for fcf in calc['projected_fcf']]  # Convert to millions
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=years,
            y=projected_fcf,
            name='Projected FCF',
            marker_color='lightblue'
        ))
        
        fig.update_layout(
            title="Projected Free Cash Flow (5-Year)",
            xaxis_title="Year",
            yaxis_title="Free Cash Flow ($M)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity analysis
        st.subheader("📊 Sensitivity Analysis")
        
        # Create sensitivity table
        growth_rates = [0.02, 0.03, 0.05, 0.07, 0.10]
        discount_rates = [0.08, 0.09, 0.10, 0.11, 0.12]
        
        base_fcf = calc['recent_fcf']
        
        sensitivity_data = []
        for discount_rate in discount_rates:
            row = []
            for growth_rate in growth_rates:
                # Simplified sensitivity calculation
                terminal_growth = 0.025
                projected_fcf_sens = [base_fcf * (1 + growth_rate) ** year for year in range(1, 6)]
                terminal_value_sens = projected_fcf_sens[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
                total_pv_sens = sum([fcf / (1 + discount_rate) ** (i + 1) for i, fcf in enumerate(projected_fcf_sens)])
                terminal_pv_sens = terminal_value_sens / (1 + discount_rate) ** 5
                enterprise_value_sens = total_pv_sens + terminal_pv_sens
                
                row.append(enterprise_value_sens / 1e9)  # Convert to billions
            sensitivity_data.append(row)
        
        sensitivity_df = pd.DataFrame(
            sensitivity_data,
            index=[f"{dr:.1%}" for dr in discount_rates],
            columns=[f"{gr:.1%}" for gr in growth_rates]
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=sensitivity_df.values,
            x=sensitivity_df.columns,
            y=sensitivity_df.index,
            colorscale='RdYlGn',
            text=sensitivity_df.values,
            texttemplate="%{text:.1f}B",
            textfont={"size": 10},
        ))
        
        fig.update_layout(
            title="Enterprise Value Sensitivity (Growth Rate vs Discount Rate)",
            xaxis_title="Growth Rate",
            yaxis_title="Discount Rate",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_asset_analysis(analysis: Dict[str, Any]):
    """Display asset-based valuation analysis"""
    
    if 'error' in analysis:
        st.error(f"Asset Analysis Error: {analysis['error']}")
        return
    
    # Display the AI analysis
    st.subheader("🤖 AI Analysis (Chain of Thought)")
    st.markdown(analysis['analysis'])
    
    if 'metrics' in analysis:
        metrics = analysis['metrics']
        
        # Key asset metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Assets", f"${metrics['total_assets']/1e9:.2f}B")
            st.metric("Book Value/Share", f"${metrics['book_value_per_share']:.2f}")
        
        with col2:
            st.metric("Total Liabilities", f"${metrics['total_liabilities']/1e9:.2f}B")
            st.metric("Shares Outstanding", f"{metrics['shares_outstanding']/1e6:.0f}M")
        
        with col3:
            st.metric("Book Value", f"${metrics['book_value']/1e9:.2f}B")
            debt_to_assets = metrics['total_liabilities'] / metrics['total_assets'] if metrics['total_assets'] > 0 else 0
            st.metric("Debt-to-Assets", f"{debt_to_assets:.2%}")
        
        # Balance sheet visualization
        fig = go.Figure(data=[
            go.Bar(name='Assets', x=['Balance Sheet'], y=[metrics['total_assets']/1e9]),
            go.Bar(name='Liabilities', x=['Balance Sheet'], y=[metrics['total_liabilities']/1e9]),
            go.Bar(name='Equity', x=['Balance Sheet'], y=[metrics['book_value']/1e9])
        ])
        
        fig.update_layout(
            title="Balance Sheet Breakdown",
            xaxis_title="",
            yaxis_title="Value ($B)",
            barmode='group',
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_final_synthesis(synthesis: Dict[str, Any]):
    """Display final synthesis and recommendation"""
    
    st.subheader("🤖 Comprehensive AI Analysis & Recommendation")
    st.markdown(synthesis['comprehensive_analysis'])
    
    # Analysis timestamp
    timestamp = synthesis.get('timestamp', datetime.now().isoformat())
    st.caption(f"Analysis completed: {timestamp}")
    
    # Create summary box
    st.info("💡 **Investment Thesis Summary**: The AI agent has analyzed this company using four distinct valuation methodologies, providing a comprehensive view of its fair value from multiple perspectives.")

# Additional utility functions
def create_charts_and_visualizations(result: Dict[str, Any]):
    """Create additional charts and visualizations"""
    
    if 'financial_data' not in result or not result['financial_data']['success']:
        return
    
    hist_data = result['financial_data']['historical_data']
    
    # Technical indicators
    st.subheader("📈 Technical Analysis")
    
    # Calculate moving averages
    hist_data['MA20'] = hist_data['Close'].rolling(window=20).mean()
    hist_data['MA50'] = hist_data['Close'].rolling(window=50).mean()
    
    # RSI calculation
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    hist_data['RSI'] = calculate_rsi(hist_data['Close'])
    
    # Create technical chart
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Price with Moving Averages', 'RSI'),
        row_heights=[0.7, 0.3]
    )
    
    # Price and moving averages
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Close'], name='Close', line=dict(color='blue')), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MA20'], name='MA20', line=dict(color='orange')), row=1, col=1)
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['MA50'], name='MA50', line=dict(color='red')), row=1, col=1)
    
    # RSI
    fig.add_trace(go.Scatter(x=hist_data.index, y=hist_data['RSI'], name='RSI', line=dict(color='purple')), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    fig.update_layout(height=600, title="Technical Analysis")
    st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()