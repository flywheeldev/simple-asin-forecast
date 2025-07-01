# simple-asin-forecast
Provide monthly ASIN unit sales history. The best of 4 time series models is chosen for each ASIN to predict the next 18 of unit sales.

🔮 ASIN Time Series Forecasting App - Enhanced AI Version
A comprehensive Streamlit application that uses advanced artificial intelligence to generate intelligent 18-month sales forecasts for multiple ASINs with automatic product lifecycle detection, seasonality recognition, and smart model selection.
🌟 What's New in This Version
🤖 Intelligent Product Classification

Automatic lifecycle detection: Growth, Mature, Decline, Introduction, Discontinued
Smart seasonality recognition: Holiday, Summer, Anti-Holiday patterns
Data-driven growth rates based on product characteristics
Discontinued product detection (4+ months of zero sales = zero forecasts)

🎯 Advanced Model Selection

Growth products → ARIMA (trend-focused forecasting)
Seasonal products → Prophet (pattern-aware with seasonality)
Mature products → Exponential Smoothing (stable forecasting)
Short data series → Simple trend extrapolation
Automatic fallback systems for robust performance

💼 Business Logic Integration

Product-specific growth rates (Growth: 12%, Decline: -3%, Introduction: 15%)
User-controlled growth rate for mature products via slider
Confidence interval estimation for forecast uncertainty
Business validation preventing unrealistic forecasts
Zero forecasts allowed for discontinued products

🎨 Enhanced User Experience

Entertaining loading messages with randomized order every run
Dark, readable colors with 22px font for better visibility
Product lifecycle dashboard showing stage distribution
Enhanced visualizations with confidence bands
Rich metadata export with business insights

🚀 Quick Start
Prerequisites
pip install streamlit pandas numpy plotly statsmodels prophet
Installation

Save the Python code as 'asin_forecasting_app.py'
Run: streamlit run asin_forecasting_app.py
Open browser to the displayed URL (typically http://localhost:8501)

📋 Data Requirements
CSV Format (Exact Column Names Required)

Date: First day of each month (Excel Short Date format supported)
ASIN: Unique product identifier
Units_Sold: Number of units sold (whole numbers)

Sample Data
Date,ASIN,Units_Sold
2023-01-01,B001,100
2023-02-01,B001,120
2023-03-01,B001,95
2023-01-01,B002,250
2023-02-01,B002,280
📄 CSV Template Download
The app includes a built-in template download button to prevent formatting errors.
🎯 How It Works
1. Smart Data Processing

Automatic date format detection (MM/DD/YYYY, YYYY-MM-DD, etc.)
Missing month auto-fill with zero sales
Complete date grid creation for consistent forecasting
Whole number conversion (no fractional units)

2. AI-Powered Analysis

Product lifecycle classification using trend analysis
Volatility assessment (coefficient of variation)
Seasonality pattern detection (Q4 holiday boost, summer patterns)
Growth trend strength calculation

3. Intelligent Model Selection
Data Length < 6 months → Simple Trend
Growth/Decline Products → ARIMA
Mature Products → Exponential Smoothing  
Seasonal Products (8+ months) → Prophet
Always Exponential Smoothing as fallback
4. Business Validation

Prevents negative forecasts (safety checks at multiple levels)
Caps unrealistic growth (max 2x recent average)
Ensures minimum viable sales (except discontinued products)
Applies seasonal adjustments (holiday/summer multipliers)

📊 Output Features
Enhanced CSV Export Includes

Historical + Forecast data combined
Product_Stage: Growth, Mature, Decline, Introduction, Discontinued
Seasonality: Holiday, Summer, Anti-Holiday, None
Growth_Rate: Applied percentage (data-driven or user-set)
Confidence: Forecast uncertainty estimate
Model_Used: AI model selected with reasoning
Validation: Business logic checks performed

Interactive Dashboard

Product lifecycle stage distribution
Seasonality pattern breakdown
Model selection summary
Top-selling ASINs prioritized in dropdown
Enhanced charts with confidence intervals

🎛️ User Controls
Growth Rate Slider (-20% to +50%)

Applied to mature products (respects business judgment)
Data-driven rates for others (Growth: 12%, Decline: -3%)
Real-time adjustment of forecasts

Automatic Processing

No buttons needed - forecasting starts on file upload
Progress tracking with entertaining messages
Session state preservation - results persist during interaction

🔬 AI Model Details
ARIMA Enhanced

Grid search optimization (p,d,q combinations)
Conservative parameter selection for stability
Best for trending data with clear direction

Exponential Smoothing Enhanced

Multiple configurations tested (trend, damped trend)
Optimized for stable/mature products
Robust performance with limited data

Prophet Enhanced

Conservative changepoint detection
Linear growth assumption
Monthly seasonality for longer series
Best for clear seasonal patterns

💡 Business Use Cases
E-commerce & Retail

Inventory Planning: 18-month demand forecasting
Budget Allocation: Resource planning by product stage
Seasonal Preparation: Holiday and summer surge planning
Discontinuation Decisions: Identify zero-growth products

Strategic Planning

Product Portfolio Analysis: Lifecycle stage distribution
Revenue Forecasting: Predictable income streams
Capacity Planning: Production and distribution alignment
Risk Management: Confidence intervals for planning

🛡️ Quality Assurance
Data Validation

Required column verification
Date format auto-detection and conversion
Numeric data type validation
Missing value handling

Forecast Quality

No negative values (multiple safety layers)
Business logic validation (realistic growth/decline limits)
Confidence estimation (historical error analysis)
Discontinued product handling (zero forecasts when appropriate)

Error Handling

Model failure graceful recovery
Individual ASIN error reporting
Alternative model selection
User-friendly error messages

📈 Expected Performance
Model Distribution (Typical)

Exponential Smoothing: 40-60% (stable products)
ARIMA: 20-30% (trending products)
Prophet: 10-20% (seasonal products with sufficient data)
Simple Trend: 5-10% (limited data products)

Accuracy Expectations

Mature products: ±15-25% confidence intervals
Growth products: ±25-40% confidence intervals
Seasonal products: ±20-30% confidence intervals
Discontinued products: 100% accuracy (zero forecasts)

🎪 Entertainment Features
Loading Messages

18 hilarious e-commerce/retail media quotes
Randomized order every app run
7-second display with large, readable fonts
Professional styling with color variety

Sample Quotes

"My conversion rate's like my gym membership—low effort, high expectations."
"Amazon said we were exclusive. Then I saw them with 6 other brands on the same shelf."
"I clicked on my own ad just to feel something."

⚠️ Limitations & Considerations
Data Requirements

Minimum: 3 months of history per ASIN
Optimal: 12+ months for seasonal detection
Prophet threshold: 8+ months for advanced seasonality

Model Assumptions

Historical patterns continue (no major market disruptions)
Linear growth trends (no exponential changes)
Seasonal patterns repeat (consistent year-over-year behavior)

Business Context Missing

No external factors: Marketing campaigns, competitive actions
No price elasticity: Price change impacts not modeled
No inventory constraints: Assumes unlimited supply capability

🔧 Troubleshooting
Common Issues
"No models succeeded"

ASIN has insufficient data (< 3 months)
All values are identical (no variation to model)
Try with longer time series

"All forecasts are zero"

Product detected as discontinued (4+ months of zero sales)
This is correct behavior for inactive products

Slow processing

Large number of ASINs increases processing time
Each ASIN tests multiple models for best fit
Consider processing in smaller batches for very large datasets

Performance Tips

Clean data first: Remove ASINs with insufficient history
Check date consistency: Ensure monthly intervals
Reasonable dataset size: Start with < 100 ASINs for testing

🎯 Business Value
Immediate Benefits

Automated forecasting replacing manual spreadsheet work
Intelligent model selection eliminating guesswork
Product lifecycle insights for strategic planning
Confidence intervals for risk assessment

Strategic Advantages

Data-driven decisions based on AI analysis
Scalable process handling hundreds of ASINs
Business logic integration ensuring realistic outputs
Professional documentation for stakeholder communication

🚀 Future Enhancement Roadmap
Advanced Features (Planned)

External data integration: Marketing spend, pricing, inventory levels
Hierarchical forecasting: Category → Brand → Product rollups
Real-time adjustments: Live data feed capabilities
Advanced metrics: Forecast value add (FVA), bias tracking

Enterprise Features

API integration for automated data feeds
Dashboard integration with BI tools
Multi-user collaboration features
Advanced alerting for forecast anomalies


Built with ❤️ using Streamlit, Statsmodels, Prophet, and Advanced AI
This app transforms basic sales history into intelligent business forecasts, combining statistical rigor with practical business sense for actionable demand planning.
🎉 Ready to forecast the future? Upload your CSV and let the AI do the work! 🎉
