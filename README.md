# 🤖 AI Agentic Financial Analysis System - Conversational Rankings

A conversational AI application that ranks S&P 500 companies based on fair market value using three different valuation methodologies. The application uses LangChain, LangGraph, and OpenAI to provide intelligent financial analysis through a natural chat interface with **transparent AI thinking** and **visual calculation cues**.

## 🚀 Features

### 💬 Conversational Interface
- **Natural Language Processing**: Ask for rankings in plain English
- **Interactive Chat**: Step-by-step guidance through the analysis process
- **Context Awareness**: Remembers your preferences and provides relevant follow-up options
- **General Question Answering**: Ask about CEOs, companies, financial concepts, and more

### 🧮 Transparent AI Thinking
- **Visual Calculation Examples**: See exactly how each valuation method works with real company data
- **Parameter Confirmation**: Review and confirm all parameters before calculations begin
- **Progress Tracking**: Real-time progress bars showing analysis of each company
- **Detailed Breakdowns**: Step-by-step calculation explanations for each valuation method

### 📊 Three Valuation Methods

#### 1. Asset Based Value
- Calculates tangible book value (Total Assets - Total Liabilities - Intangible Assets - Goodwill)
- Uses conservative estimates to account for asset quality
- Applies 80% discount to book value as safety margin
- **Best for**: Asset-heavy businesses (utilities, real estate, manufacturing)

#### 2. Discounted Cash Flow (DCF) Value
- Projects future free cash flows for 5 years
- Calculates terminal value using perpetual growth model
- Discounts all cash flows to present value using WACC
- **Key Assumptions**:
  - 5% annual growth rate for first 5 years
  - 2.5% terminal growth rate
  - Beta of 1.0 (market average)
  - WACC = Risk-free rate + Market risk premium
- **Best for**: Companies with stable cash flows and predictable growth

#### 3. Comparable Company Analysis
- Uses industry average multiples to value companies
- Combines EV/Revenue, P/E, and P/B ratios
- Averages multiple approaches for balanced valuation
- **Key Assumptions**:
  - EV/Revenue multiple: 2.0x
  - P/E multiple: 15.0x
  - P/B multiple: 1.5x
- **Best for**: Companies with comparable peers in the same industry

### 🎯 AI-Powered Analysis
- **Intelligent Insights**: AI agent provides comprehensive analysis of rankings
- **Sector Analysis**: Identifies trends and patterns across industries
- **Risk Assessment**: Highlights potential limitations and considerations
- **Investment Implications**: Provides actionable investment insights

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- OpenAI API key
- Optional: LangSmith API key for tracing

### Setup
1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd v5
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare data**
   - Ensure `sp500_financial_data.csv` is in the project directory
   - The CSV should contain comprehensive financial data for S&P 500 companies

4. **Set up API keys**
   - Get an OpenAI API key from [OpenAI Platform](https://platform.openai.com/)
   - Optional: Get a LangSmith API key for tracing and monitoring

## 🚀 Usage

### Starting the Application
```bash
streamlit run main.py
```

### Enhanced Conversational Flow

#### 1. **Initial Request**
You can start with either:

**Company Rankings:**
```
"I want to see company rankings based on fair market value"
"Show me rankings of companies"
"Rank companies by value"
```

**General Questions:**
```
"Who is the CEO of Apple?"
"What is DCF valuation?"
"Explain P/E ratio"
"Tell me about Warren Buffett"
```

#### 2. **Select Valuation Method**
The AI will ask you to choose a valuation method:
- **Asset Based Value**: For asset-heavy companies
- **Discounted Cash Flow Value**: For companies with stable cash flows
- **Comparable Company Analysis**: For peer-based valuation

#### 3. **🔍 Parameter Confirmation & Visual Examples**
**NEW!** Before starting calculations, the AI will:
- Show a **sample calculation** using a well-known company (AAPL, MSFT, or GOOGL)
- Display **detailed breakdown** of the valuation method
- Confirm all **parameters and assumptions**
- Show **expected processing time** and data quality metrics

#### 4. **📊 Real-Time Progress Tracking**
**NEW!** During calculations, you'll see:
- **Progress bar** showing completion percentage
- **Live status updates** for each company being analyzed
- **Company-by-company processing** with ticker and name display
- **Completion confirmation** with sorting status

#### 5. **Review Results**
The application will show:
- Top 20 ranked companies with key metrics
- Interactive visualizations (sector distribution, fair value vs revenue)
- AI-generated analysis and insights
- Key assumptions and limitations

#### 6. **Follow-up Questions**
Ask for more details:
```
"Explain the assumptions"
"Show sector analysis"
"Start a new analysis"
```

## 📊 Sample Conversation with Visual Cues

```
User: I want to see company rankings based on fair market value

AI: I'd be happy to help you rank companies based on fair market value!

I can calculate rankings using three different valuation methods:

1. Asset Based Value - Based on tangible book value and asset quality
2. Discounted Cash Flow Value - Based on projected future cash flows (requires macroeconomic parameters)
3. Comparable Company Analysis - Based on industry multiples and peer comparisons

Which valuation method would you prefer to use?

User: Discounted Cash Flow Value

AI: Great choice! For Discounted Cash Flow analysis, I need some macroeconomic parameters to calculate the discount rate.

Please provide:
1. Risk-free rate (e.g., 4.5% for current 10-year Treasury)
2. Market Risk Premium (e.g., 6% for typical equity risk premium)

You can provide these as percentages (e.g., "4.5% and 6%") or as decimals (e.g., "0.045 and 0.06").

User: 4.5% and 6%

AI: Perfect! I have your parameters:
- Risk-free Rate: 4.50%
- Market Risk Premium: 6.00%

Let me show you how the DCF calculation works and confirm everything before we start analyzing all 503 companies.

[VISUAL EXAMPLE APPEARS]
🧮 Sample Calculation
Example calculation for AAPL - Apple Inc.

Key Parameters:
- Base FCF: $23,952M
- Risk-free Rate: 4.5%
- Market Risk Premium: 6.0%
- WACC: 10.5%
- Growth Rate: 5.0%

Projected Cash Flows:
- Year 1: $25,150M
- Year 2: $26,407M
- Year 3: $27,728M
- Year 4: $29,114M
- Year 5: $30,570M
- Terminal Value: $394.76B
Final Enterprise Value: $394.76B

[PARAMETER CONFIRMATION APPEARS]
🔍 Parameter Confirmation
Valuation Method: Discounted Cash Flow Value

Methodology:
- Projects free cash flows for 5 years with 4.5% risk-free rate
- Uses 6.0% market risk premium for discount rate calculation
- Applies 5% growth rate for first 5 years, 2.5% terminal growth
- Discounts all cash flows to present value using WACC

Companies to Analyze: 503
Data Quality: High
Processing Time: ~30 seconds

[START ANALYSIS BUTTON]

User: [Clicks "🚀 Start Analysis"]

[PROGRESS TRACKING APPEARS]
Analyzing ABBV - AbbVie (1/503) [████████░░] 20%
Analyzing MMM - 3M (2/503) [████████░░] 20%
...
✅ Analysis complete! Sorting results...

[RESULTS APPEAR]
📈 Rankings Results
Top 20 Companies by Discounted Cash Flow Value
[Interactive table and visualizations]
```

## 📈 Output Features

### Rankings Table
- Company ticker and name
- Sector classification
- Fair value in billions
- Revenue and net income
- Interactive sorting and filtering

### Visualizations
- **Sector Distribution Pie Chart**: Shows industry breakdown of top companies
- **Fair Value vs Revenue Scatter Plot**: Identifies value opportunities
- **Interactive Charts**: Hover for detailed information

### AI Analysis
- **Key Insights**: Top-ranked companies analysis
- **Sector Trends**: Industry distribution patterns
- **Methodology Considerations**: Valuation approach strengths and limitations
- **Risk Assessment**: Potential risks and uncertainties
- **Investment Implications**: Actionable investment insights

## 🔧 Configuration

### API Keys
- **OpenAI API Key**: Required for AI analysis
- **LangSmith API Key**: Optional for tracing and monitoring

### Advanced Parameters
- **Risk-free Rate**: Default 4.5% (10-year Treasury)
- **Market Risk Premium**: Default 6% (equity risk premium)
- **Growth Assumptions**: 5% for 5 years, 2.5% terminal
- **Valuation Multiples**: Conservative industry averages

## 🧪 Testing

Run the test suite to verify functionality:
```bash
python test_conversational.py
```

The test suite validates:
- Data loading and processing
- Valuation function calculations
- Rankings generation
- Conversational flow logic

## 📋 Data Requirements

The application requires `sp500_financial_data.csv` with the following columns:
- `ticker`: Company stock symbol
- `company_name`: Full company name
- `sector`: Industry sector
- `revenues`: Total revenue
- `net_income_loss`: Net income
- `assets`: Total assets
- `liabilities`: Total liabilities
- `free_cash_flow`: Free cash flow
- `book_value`: Book value of equity
- Additional financial metrics for comprehensive analysis

## 🎯 Use Cases

## 💬 Types of Questions You Can Ask

### Company Information
- **Leadership**: "Who is the CEO of Apple?" or "Who founded Tesla?"
- **Company Details**: "Where is Microsoft headquartered?" or "When was Amazon founded?"
- **Business Model**: "What does Apple do?" or "How does Netflix make money?"

### Financial Concepts
- **Valuation Methods**: "What is DCF valuation?" or "Explain P/E ratio"
- **Financial Metrics**: "What is market cap?" or "Define beta in finance"
- **Investment Terms**: "What does IPO mean?" or "Explain dividend yield"

### Market Analysis
- **Industry Trends**: "What is happening in the tech sector?" or "Compare Apple vs Microsoft"
- **Investment Strategies**: "What is value investing?" or "Explain growth vs value stocks"
- **Market Dynamics**: "What is market risk premium?" or "How do interest rates affect stocks?"

### General Finance
- **Portfolio Management**: "What is diversification?" or "How to build a portfolio?"
- **Risk Management**: "What is systematic risk?" or "How to measure investment risk?"
- **Economic Concepts**: "What is inflation?" or "How does GDP affect markets?"

### Investment Research
- Identify undervalued companies across different sectors
- Compare valuation approaches for the same company
- Understand sector-specific valuation trends

### Portfolio Analysis
- Rank potential investment candidates
- Assess relative value across different industries
- Identify value opportunities in specific sectors

### Financial Education
- Learn about different valuation methodologies
- Understand the impact of macroeconomic factors
- Explore the relationship between fundamentals and fair value

### General Knowledge & Research
- Get information about company executives and leadership
- Learn about financial concepts and terminology
- Research company history and business models
- Understand investment strategies and market dynamics

## 🔍 Limitations and Considerations

### Valuation Method Limitations
- **Asset Based**: May undervalue companies with strong intangible assets
- **DCF**: Highly sensitive to growth and discount rate assumptions
- **Comparable**: Assumes companies are comparable within sectors

### Data Limitations
- Based on historical financial data
- May not reflect current market conditions
- Assumes data quality and completeness

### Investment Disclaimer
This application is for educational and research purposes only. It does not constitute investment advice. Always conduct thorough due diligence before making investment decisions.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the test suite: `python test_conversational.py`
2. Verify data file format and completeness
3. Ensure API keys are correctly configured
4. Review the conversational flow examples

## 🚀 Future Enhancements

- [ ] Real-time market data integration
- [ ] Additional valuation methods (LBO, Sum of Parts)
- [ ] Custom parameter tuning
- [ ] Export functionality for rankings
- [ ] Historical analysis and backtesting
- [ ] Multi-language support
- [ ] Mobile-responsive interface
- [ ] Advanced visualization options
- [ ] Custom calculation parameters
- [ ] Batch analysis capabilities 