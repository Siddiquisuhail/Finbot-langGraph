# config.py
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Configuration class for the financial analysis system"""
    
    # API Configuration
    OPENAI_API_KEY: Optional[str] = None
    LANGSMITH_API_KEY: Optional[str] = None
    
    # Model Configuration
    OPENAI_MODEL: str = "gpt-4o"
    TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 4000
    
    # Analysis Configuration
    DEFAULT_RISK_FREE_RATE: float = 0.045
    DEFAULT_MARKET_RISK_PREMIUM: float = 0.06
    DEFAULT_TERMINAL_GROWTH_RATE: float = 0.025
    
    # Data Configuration
    MAX_COMPARABLE_COMPANIES: int = 10
    DEFAULT_ANALYSIS_PERIOD: str = "5y"
    
    # LangSmith Configuration
    LANGCHAIN_TRACING_V2: str = "true"
    LANGCHAIN_PROJECT: str = "financial-analysis-agent"
    
    @classmethod
    def from_env(cls) -> 'Config':
        """Load configuration from environment variables"""
        return cls(
            OPENAI_API_KEY=os.getenv('OPENAI_API_KEY'),
            LANGSMITH_API_KEY=os.getenv('LANGSMITH_API_KEY'),
            LANGCHAIN_TRACING_V2=os.getenv('LANGCHAIN_TRACING_V2', 'true'),
            LANGCHAIN_PROJECT=os.getenv('LANGCHAIN_PROJECT', 'financial-analysis-agent')
        )

# utils.py
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import yfinance as yf
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinancialCalculator:
    """Utility class for financial calculations"""
    
    @staticmethod
    def calculate_wacc(risk_free_rate: float, beta: float, market_risk_premium: float, 
                      debt_ratio: float = 0.0, tax_rate: float = 0.21) -> float:
        """Calculate Weighted Average Cost of Capital"""
        cost_of_equity = risk_free_rate + beta * market_risk_premium
        cost_of_debt = risk_free_rate + 0.02  # Simplified assumption
        
        wacc = (1 - debt_ratio) * cost_of_equity + debt_ratio * cost_of_debt * (1 - tax_rate)
        return wacc
    
    @staticmethod
    def calculate_dcf_value(cash_flows: List[float], terminal_value: float, 
                           discount_rate: float) -> Tuple[float, List[float]]:
        """Calculate DCF present value"""
        present_values = []
        
        # Discount cash flows
        for i, cf in enumerate(cash_flows):
            pv = cf / (1 + discount_rate) ** (i + 1)
            present_values.append(pv)
        
        # Discount terminal value
        terminal_pv = terminal_value / (1 + discount_rate) ** len(cash_flows)
        
        total_pv = sum(present_values) + terminal_pv
        
        return total_pv, present_values
    
    @staticmethod
    def calculate_terminal_value(final_cash_flow: float, terminal_growth_rate: float,
                                discount_rate: float) -> float:
        """Calculate terminal value using Gordon Growth Model"""
        return final_cash_flow * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    
    @staticmethod
    def calculate_financial_ratios(financials: pd.DataFrame, balance_sheet: pd.DataFrame,
                                 info: Dict[str, Any]) -> Dict[str, float]:
        """Calculate key financial ratios"""
        ratios = {}
        
        try:
            # Profitability ratios
            if not financials.empty and 'Total Revenue' in financials.index:
                revenue = financials.loc['Total Revenue'].iloc[0]
                net_income = financials.loc['Net Income'].iloc[0] if 'Net Income' in financials.index else 0
                ratios['profit_margin'] = net_income / revenue if revenue != 0 else 0
            
            # Liquidity ratios
            if not balance_sheet.empty:
                current_assets = balance_sheet.loc['Current Assets'].iloc[0] if 'Current Assets' in balance_sheet.index else 0
                current_liabilities = balance_sheet.loc['Current Liabilities'].iloc[0] if 'Current Liabilities' in balance_sheet.index else 0
                ratios['current_ratio'] = current_assets / current_liabilities if current_liabilities != 0 else 0
            
            # Market ratios from info
            ratios['pe_ratio'] = info.get('trailingPE', 0)
            ratios['pb_ratio'] = info.get('priceToBook', 0)
            ratios['ps_ratio'] = info.get('priceToSalesTrailing12Months', 0)
            ratios['peg_ratio'] = info.get('pegRatio', 0)
            
        except Exception as e:
            logger.warning(f"Error calculating ratios: {e}")
        
        return ratios
    
    @staticmethod
    def calculate_volatility_metrics(price_data: pd.Series) -> Dict[str, float]:
        """Calculate volatility and risk metrics"""
        returns = price_data.pct_change().dropna()
        
        metrics = {
            'daily_volatility': returns.std(),
            'annualized_volatility': returns.std() * np.sqrt(252),
            'downside_deviation': returns[returns < 0].std() * np.sqrt(252),
            'max_drawdown': (price_data / price_data.cummax() - 1).min(),
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0,
            'var_95': returns.quantile(0.05),
            'skewness': returns.skew(),
            'kurtosis': returns.kurtosis()
        }
        
        return metrics

class DataValidator:
    """Utility class for data validation and cleaning"""
    
    @staticmethod
    def validate_ticker(ticker: str) -> Tuple[bool, str]:
        """Validate stock ticker"""
        if not ticker or not isinstance(ticker, str):
            return False, "Ticker must be a non-empty string"
        
        ticker = ticker.upper().strip()
        
        # Basic validation
        if len(ticker) > 10:
            return False, "Ticker too long"
        
        if not ticker.replace('.', '').isalnum():
            return False, "Ticker contains invalid characters"
        
        try:
            # Test if ticker exists
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or 'symbol' not in info:
                return False, "Ticker not found"
            
            return True, ticker
            
        except Exception as e:
            return False, f"Error validating ticker: {str(e)}"
    
    @staticmethod
    def clean_financial_data(data: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize financial data"""
        if data.empty:
            return data
        
        # Remove columns with all NaN values
        data = data.dropna(axis=1, how='all')
        
        # Fill NaN values with 0 for calculation purposes
        data = data.fillna(0)
        
        return data
    
    @staticmethod
    def validate_analysis_parameters(params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate analysis parameters"""
        errors = []
        
        # Validate duration
        valid_durations = ["1 Month", "3 Months", "6 Months", "1 Year", "2 Years", "5 Years"]
        if params.get('duration') not in valid_durations:
            errors.append(f"Invalid duration. Must be one of: {valid_durations}")
        
        # Validate rates
        if not 0 <= params.get('risk_free_rate', 0) <= 1:
            errors.append("Risk-free rate must be between 0 and 1")
        
        if not 0 <= params.get('market_risk_premium', 0) <= 1:
            errors.append("Market risk premium must be between 0 and 1")
        
        return len(errors) == 0, errors

class CompanyScreener:
    """Utility class for finding comparable companies"""
    
    # Industry mapping for comparable company selection
    INDUSTRY_SECTORS = {
        'Technology': {
            'Software': ['MSFT', 'ORCL', 'CRM', 'ADBE', 'INTU', 'VMW'],
            'Hardware': ['AAPL', 'HPQ', 'DELL', 'WDC', 'STX'],
            'Semiconductors': ['NVDA', 'AMD', 'INTC', 'QCOM', 'TXN', 'AVGO'],
            'Internet': ['GOOGL', 'META', 'AMZN', 'NFLX', 'UBER', 'LYFT']
        },
        'Healthcare': {
            'Pharmaceuticals': ['PFE', 'JNJ', 'MRK', 'ABBV', 'BMY', 'LLY'],
            'Biotechnology': ['GILD', 'AMGN', 'BIIB', 'REGN', 'CELG'],
            'Medical Devices': ['MDT', 'ABT', 'TMO', 'DHR', 'SYK'],
            'Health Insurance': ['UNH', 'ANTM', 'CVS', 'CI', 'HUM']
        },
        'Financial Services': {
            'Banks': ['JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC'],
            'Investment Banking': ['GS', 'MS', 'BLK', 'SCHW'],
            'Insurance': ['BRK-B', 'AIG', 'TRV', 'PGR', 'ALL'],
            'Credit Cards': ['V', 'MA', 'AXP', 'COF']
        },
        'Consumer Cyclical': {
            'Retail': ['AMZN', 'WMT', 'TGT', 'HD', 'LOW', 'COST'],
            'Restaurants': ['MCD', 'SBUX', 'YUM', 'CMG', 'DRI'],
            'Automotive': ['TSLA', 'F', 'GM', 'TM', 'HMC'],
            'Apparel': ['NKE', 'ADDYY', 'VFC', 'PVH', 'RL']
        },
        'Energy': {
            'Oil & Gas': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'HAL'],
            'Renewable Energy': ['NEE', 'DUK', 'SO', 'AEP', 'EXC'],
            'Oil Services': ['SLB', 'HAL', 'BKR', 'NOV', 'FTI']
        },
        'Industrials': {
            'Aerospace': ['BA', 'LMT', 'RTX', 'NOC', 'GD'],
            'Manufacturing': ['CAT', 'DE', 'MMM', 'GE', 'EMR'],
            'Transportation': ['UNP', 'CSX', 'NSC', 'FDX', 'UPS'],
            'Construction': ['DHI', 'LEN', 'NVR', 'PHM', 'TOL']
        },
        'Consumer Staples': {
            'Food & Beverages': ['KO', 'PEP', 'KHC', 'MDLZ', 'GIS'],
            'Household Products': ['PG', 'UL', 'CL', 'KMB', 'CHD'],
            'Tobacco': ['MO', 'PM', 'BTI'],
            'Retail Food': ['WMT', 'COST', 'KR', 'SYY']
        },
        'Utilities': {
            'Electric': ['NEE', 'DUK', 'SO', 'AEP', 'EXC'],
            'Gas': ['SRE', 'PEG', 'D', 'PCG'],
            'Water': ['AWK', 'WTR', 'CWCO']
        }
    }
    
    @staticmethod
    def find_comparables(ticker: str, sector: str, industry: str, 
                        market_cap: float = None, limit: int = 5) -> List[str]:
        """Find comparable companies based on sector and industry"""
        
        comparables = []
        
        # First try to find exact industry match
        if sector in CompanyScreener.INDUSTRY_SECTORS:
            sector_companies = CompanyScreener.INDUSTRY_SECTORS[sector]
            
            # Look for industry-specific matches
            for ind, companies in sector_companies.items():
                if industry.lower() in ind.lower() or ind.lower() in industry.lower():
                    comparables.extend(companies)
                    break
            
            # If no industry match, use all companies in sector
            if not comparables:
                for companies in sector_companies.values():
                    comparables.extend(companies)
        
        # Remove the target ticker itself
        comparables = [c for c in comparables if c != ticker.upper()]
        
        # If market cap is provided, try to filter by size
        if market_cap and comparables:
            filtered_comparables = []
            for comp_ticker in comparables:
                try:
                    comp_stock = yf.Ticker(comp_ticker)
                    comp_info = comp_stock.info
                    comp_market_cap = comp_info.get('marketCap', 0)
                    
                    # Include if within 50% to 200% of target market cap
                    if comp_market_cap and 0.5 * market_cap <= comp_market_cap <= 2.0 * market_cap:
                        filtered_comparables.append(comp_ticker)
                except:
                    continue
            
            if filtered_comparables:
                comparables = filtered_comparables
        
        return comparables[:limit]

class TechnicalAnalysis:
    """Utility class for technical analysis calculations"""
    
    @staticmethod
    def calculate_moving_averages(prices: pd.Series, windows: List[int] = [20, 50, 200]) -> pd.DataFrame:
        """Calculate multiple moving averages"""
        ma_data = pd.DataFrame(index=prices.index)
        
        for window in windows:
            ma_data[f'MA_{window}'] = prices.rolling(window=window).mean()
        
        return ma_data
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_bollinger_bands(prices: pd.Series, window: int = 20, 
                                 num_std: float = 2) -> pd.DataFrame:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        bands = pd.DataFrame(index=prices.index)
        bands['middle'] = ma
        bands['upper'] = ma + (std * num_std)
        bands['lower'] = ma - (std * num_std)
        
        return bands
    
    @staticmethod
    def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, 
                      signal: int = 9) -> pd.DataFrame:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd_line = exp1 - exp2
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        macd_data = pd.DataFrame(index=prices.index)
        macd_data['macd'] = macd_line
        macd_data['signal'] = signal_line
        macd_data['histogram'] = histogram
        
        return macd_data
    
    @staticmethod
    def identify_support_resistance(prices: pd.Series, window: int = 20) -> Dict[str, float]:
        """Identify support and resistance levels"""
        # Simple approach using rolling min/max
        recent_data = prices.tail(window * 5)  # Look at last 100 periods
        
        # Find local minima and maxima
        highs = recent_data.rolling(window=window, center=True).max()
        lows = recent_data.rolling(window=window, center=True).min()
        
        # Identify levels where price touched multiple times
        resistance_levels = highs[highs == recent_data].unique()
        support_levels = lows[lows == recent_data].unique()
        
        return {
            'resistance': float(np.median(resistance_levels)) if len(resistance_levels) > 0 else float(recent_data.max()),
            'support': float(np.median(support_levels)) if len(support_levels) > 0 else float(recent_data.min())
        }

class ReportGenerator:
    """Utility class for generating formatted reports"""
    
    @staticmethod
    def format_currency(amount: float, in_millions: bool = False) -> str:
        """Format currency values"""
        if in_millions:
            return f"${amount/1e6:.1f}M"
        elif abs(amount) >= 1e9:
            return f"${amount/1e9:.2f}B"
        elif abs(amount) >= 1e6:
            return f"${amount/1e6:.1f}M"
        elif abs(amount) >= 1e3:
            return f"${amount/1e3:.1f}K"
        else:
            return f"${amount:.2f}"
    
    @staticmethod
    def format_percentage(value: float, decimal_places: int = 2) -> str:
        """Format percentage values"""
        return f"{value * 100:.{decimal_places}f}%"
    
    @staticmethod
    def format_ratio(value: float, decimal_places: int = 2) -> str:
        """Format ratio values"""
        if value == 0 or np.isnan(value):
            return "N/A"
        return f"{value:.{decimal_places}f}"
    
    @staticmethod
    def generate_executive_summary(analysis_results: Dict[str, Any]) -> str:
        """Generate executive summary from analysis results"""
        
        ticker = analysis_results.get('ticker', 'Unknown')
        
        # Extract key metrics
        current_price = 0
        market_cap = 0
        
        if 'financial_data' in analysis_results:
            info = analysis_results['financial_data'].get('info', {})
            current_price = info.get('currentPrice', 0)
            market_cap = info.get('marketCap', 0)
        
        summary = f"""
        ## Executive Summary for {ticker}
        
        **Current Market Position:**
        - Current Price: {ReportGenerator.format_currency(current_price)}
        - Market Capitalization: {ReportGenerator.format_currency(market_cap)}
        
        **Valuation Analysis Summary:**
        """
        
        # Add valuation method summaries
        if 'market_price_analysis' in analysis_results:
            summary += "\n- **Market Price Method**: Analysis based on current market multiples and trading patterns"
        
        if 'comparable_analysis' in analysis_results:
            summary += "\n- **Comparable Company Analysis**: Valuation relative to industry peers"
        
        if 'dcf_analysis' in analysis_results:
            dcf_data = analysis_results['dcf_analysis']
            if 'calculations' in dcf_data:
                fair_value = dcf_data['calculations'].get('fair_value_per_share', 0)
                summary += f"\n- **DCF Analysis**: Intrinsic value of {ReportGenerator.format_currency(fair_value)} per share"
        
        if 'asset_analysis' in analysis_results:
            asset_data = analysis_results['asset_analysis']
            if 'metrics' in asset_data:
                book_value = asset_data['metrics'].get('book_value_per_share', 0)
                summary += f"\n- **Asset-Based Valuation**: Book value of {ReportGenerator.format_currency(book_value)} per share"
        
        return summary
    
    @staticmethod
    def create_risk_assessment(analysis_results: Dict[str, Any]) -> Dict[str, str]:
        """Create risk assessment based on analysis results"""
        
        risks = {
            'market_risk': 'Medium',
            'liquidity_risk': 'Low',
            'operational_risk': 'Medium',
            'financial_risk': 'Medium'
        }
        
        # Adjust risks based on analysis results
        if 'market_price_analysis' in analysis_results:
            metrics = analysis_results['market_price_analysis'].get('metrics', {})
            volatility = metrics.get('volatility', 0)
            
            if volatility > 0.4:  # 40% annual volatility
                risks['market_risk'] = 'High'
            elif volatility < 0.2:  # 20% annual volatility
                risks['market_risk'] = 'Low'
        
        return risks

# Advanced analysis utilities
class AdvancedAnalytics:
    """Advanced analytical utilities for financial analysis"""
    
    @staticmethod
    def calculate_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        try:
            # Align the series
            aligned_data = pd.concat([stock_returns, market_returns], axis=1).dropna()
            if len(aligned_data) < 20:  # Need sufficient data points
                return 1.0
            
            stock_aligned = aligned_data.iloc[:, 0]
            market_aligned = aligned_data.iloc[:, 1]
            
            # Calculate beta
            covariance = np.cov(stock_aligned, market_aligned)[0][1]
            market_variance = np.var(market_aligned)
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception:
            return 1.0  # Default beta if calculation fails
    
    @staticmethod
    def monte_carlo_simulation(initial_price: float, mu: float, sigma: float,
                              days: int = 252, simulations: int = 1000) -> Dict[str, Any]:
        """Run Monte Carlo simulation for price prediction"""
        
        # Generate random price paths
        dt = 1/252  # Daily time step
        price_paths = np.zeros((simulations, days))
        price_paths[:, 0] = initial_price
        
        for t in range(1, days):
            random_shocks = np.random.normal(0, 1, simulations)
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * random_shocks
            )
        
        final_prices = price_paths[:, -1]
        
        results = {
            'mean_final_price': np.mean(final_prices),
            'median_final_price': np.median(final_prices),
            'std_final_price': np.std(final_prices),
            'percentile_5': np.percentile(final_prices, 5),
            'percentile_95': np.percentile(final_prices, 95),
            'probability_profit': np.sum(final_prices > initial_price) / simulations,
            'max_price': np.max(final_prices),
            'min_price': np.min(final_prices),
            'price_paths': price_paths
        }
        
        return results
    
    @staticmethod
    def value_at_risk(returns: pd.Series, confidence_level: float = 0.05) -> Dict[str, float]:
        """Calculate Value at Risk (VaR) and Expected Shortfall (ES)"""
        
        # Sort returns
        sorted_returns = returns.sort_values()
        
        # Calculate VaR
        var_index = int(confidence_level * len(sorted_returns))
        var = sorted_returns.iloc[var_index]
        
        # Calculate Expected Shortfall (Conditional VaR)
        es = sorted_returns.iloc[:var_index].mean()
        
        return {
            'var_95': var,
            'expected_shortfall': es,
            'worst_return': sorted_returns.min(),
            'best_return': sorted_returns.max()
        }
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> float:
        """Calculate Information Ratio"""
        try:
            active_returns = portfolio_returns - benchmark_returns
            tracking_error = active_returns.std()
            
            if tracking_error == 0:
                return 0.0
            
            information_ratio = active_returns.mean() / tracking_error
            return information_ratio
            
        except Exception:
            return 0.0

# Error handling and logging utilities
class FinancialAnalysisError(Exception):
    """Custom exception for financial analysis errors"""
    pass

class DataQualityError(FinancialAnalysisError):
    """Exception for data quality issues"""
    pass

class CalculationError(FinancialAnalysisError):
    """Exception for calculation errors"""
    pass

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration"""
    
    # Create logger
    logger = logging.getLogger('financial_analysis')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    if not logger.handlers:
        logger.addHandler(console_handler)
    
    return logger