import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Plotly (with error handling for installation)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly not installed. Please install with: pip install plotly")
    st.stop()

# Time series modeling libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    import itertools
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    st.error("Statsmodels not installed. Please install with: pip install statsmodels")
    st.stop()

# Prophet (with error handling for installation)
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

def load_and_validate_data(uploaded_file):
    """Load and validate the uploaded CSV file"""
    try:
        df = pd.read_csv(uploaded_file)
        
        # Validate required columns
        required_columns = ['Date', 'ASIN', 'Units_Sold']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            return None
        
        # Convert Date column to datetime with multiple format attempts
        date_formats = [
            None,  # Let pandas auto-detect
            '%m/%d/%Y',  # 7/1/2024
            '%m/%d/%y',  # 7/1/24
            '%Y-%m-%d',  # 2024-07-01
            '%d/%m/%Y',  # 1/7/2024 (day first)
            '%Y/%m/%d',  # 2024/7/1
        ]
        
        date_parsed = False
        for fmt in date_formats:
            try:
                if fmt is None:
                    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
                else:
                    df['Date'] = pd.to_datetime(df['Date'], format=fmt, errors='coerce')
                
                # Check if parsing was successful (no NaT values)
                if not df['Date'].isna().any():
                    date_parsed = True
                    break
                    
            except Exception as e:
                continue
        
        # If no format worked perfectly, try the most permissive approach
        if not date_parsed:
            try:
                df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True, errors='coerce')
                # Remove rows where date parsing failed
                df = df.dropna(subset=['Date'])
                if len(df) == 0:
                    st.error("No valid dates could be parsed from your file.")
                    return None
                elif df['Date'].isna().any():
                    st.warning(f"Some dates could not be parsed and were removed. {df['Date'].isna().sum()} rows affected.")
                date_parsed = True
            except:
                pass
        
        if not date_parsed:
            st.error("Could not parse dates. Please ensure dates are in a recognizable format (e.g., 2024-07-01, 7/1/2024, etc.)")
            st.info("Example of acceptable date formats: 2024-01-01, 1/1/2024, 01/01/2024")
            return None
        
        # Validate data types and round Units_Sold to whole numbers
        if not pd.api.types.is_numeric_dtype(df['Units_Sold']):
            st.error("Units_Sold column must contain numeric values")
            return None
        
        # Round Units_Sold to whole numbers and convert to int
        df['Units_Sold'] = df['Units_Sold'].fillna(0).round().astype(int)
        
        # Create complete date range for each ASIN (fill missing months with 0)
        all_dates = pd.date_range(start=df['Date'].min(), end=df['Date'].max(), freq='MS')
        all_asins = df['ASIN'].unique()
        
        # Create complete grid and merge with actual data
        complete_grid = pd.MultiIndex.from_product([all_asins, all_dates], names=['ASIN', 'Date']).to_frame(index=False)
        df_complete = complete_grid.merge(df, on=['ASIN', 'Date'], how='left')
        df_complete['Units_Sold'] = df_complete['Units_Sold'].fillna(0).astype(int)
        
        # Sort by ASIN and Date
        df_complete = df_complete.sort_values(['ASIN', 'Date'])
        
        return df_complete
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

def analyze_product_characteristics(ts):
    """Analyze product lifecycle stage and characteristics from sales data only"""
    values = ts.values
    dates = ts.index
    
    if len(values) < 3:
        return {"stage": "unknown", "growth_rate": 0.05, "seasonality": "none", "volatility": "high"}
    
    # Calculate basic metrics
    recent_period = min(6, len(values))
    early_period = min(6, len(values))
    
    recent_avg = np.mean(values[-recent_period:]) if recent_period > 0 else 0
    early_avg = np.mean(values[:early_period]) if early_period > 0 else recent_avg
    
    # Trend calculation
    if len(values) >= 4:
        # Use linear regression for trend
        x = np.arange(len(values))
        trend_slope = np.polyfit(x, values, 1)[0]
        trend_strength = abs(trend_slope) / (np.mean(values) + 1)
    else:
        trend_slope = 0
        trend_strength = 0
    
    # Volatility (coefficient of variation)
    cv = np.std(values) / (np.mean(values) + 1)
    
    # Seasonality detection (basic)
    seasonality_type = detect_seasonality_pattern(values, dates)
    
    # Lifecycle stage classification
    stage, growth_rate = classify_product_stage(
        values, trend_slope, trend_strength, cv, recent_avg, early_avg
    )
    
    return {
        "stage": stage,
        "growth_rate": growth_rate,
        "seasonality": seasonality_type,
        "volatility": "high" if cv > 0.5 else "medium" if cv > 0.3 else "low",
        "trend_slope": trend_slope,
        "recent_avg": recent_avg,
        "early_avg": early_avg
    }

def detect_seasonality_pattern(values, dates):
    """Detect seasonality patterns from sales data"""
    if len(values) < 6:
        return "none"
    
    try:
        # Convert dates to months for seasonality detection
        months = [d.month for d in dates]
        
        # Check for holiday seasonality (Q4: Oct, Nov, Dec)
        q4_months = [10, 11, 12]
        q4_sales = [values[i] for i, month in enumerate(months) if month in q4_months]
        non_q4_sales = [values[i] for i, month in enumerate(months) if month not in q4_months]
        
        if len(q4_sales) >= 2 and len(non_q4_sales) >= 2:
            q4_avg = np.mean(q4_sales)
            non_q4_avg = np.mean(non_q4_sales)
            
            if q4_avg > non_q4_avg * 1.4:  # 40% boost in Q4
                return "holiday"
            elif q4_avg < non_q4_avg * 0.7:  # 30% drop in Q4
                return "anti_holiday"
        
        # Check for summer seasonality (Jun, Jul, Aug)
        summer_months = [6, 7, 8]
        summer_sales = [values[i] for i, month in enumerate(months) if month in summer_months]
        non_summer_sales = [values[i] for i, month in enumerate(months) if month not in summer_months]
        
        if len(summer_sales) >= 2 and len(non_summer_sales) >= 2:
            summer_avg = np.mean(summer_sales)
            non_summer_avg = np.mean(non_summer_sales)
            
            if summer_avg > non_summer_avg * 1.3:
                return "summer"
        
        return "none"
        
    except:
        return "none"

def classify_product_stage(values, trend_slope, trend_strength, cv, recent_avg, early_avg):
    """Classify product lifecycle stage and assign appropriate growth rate"""
    
    # Check for discontinued products (zero sales in last 4+ months)
    if len(values) >= 4:
        last_four_months = values[-4:]
        if all(v == 0 for v in last_four_months):
            return "discontinued", 0.0  # No growth for discontinued products
    
    # Strong upward trend = Growth stage
    if trend_slope > 0 and trend_strength > 0.1 and recent_avg > early_avg * 1.2:
        return "growth", 0.12  # 12% annual growth
    
    # Strong downward trend = Decline stage (but cap decline to prevent negative forecasts)
    elif trend_slope < 0 and trend_strength > 0.1 and recent_avg < early_avg * 0.8:
        return "decline", -0.03  # Reduced decline rate: 3% annual decline (was 8%)
    
    # High volatility but no clear trend = Introduction/Testing
    elif cv > 0.6 and len(values) <= 8:
        return "introduction", 0.15  # 15% growth (optimistic for new products)
    
    # Low volatility, stable sales = Mature
    elif cv < 0.3 and abs(trend_slope) < 1:
        return "mature", 0.03  # 3% mature growth
    
    # Moderate growth = Early maturity
    elif trend_slope > 0 and trend_strength > 0.05:
        return "early_mature", 0.08  # 8% growth
    
    # Default to mature with standard growth
    else:
        return "mature", 0.05  # 5% standard growth

def apply_seasonal_adjustments(forecast_values, seasonality_type, start_month):
    """Apply seasonal patterns to forecasts"""
    if seasonality_type == "none":
        return forecast_values
    
    seasonal_multipliers = get_seasonal_multipliers(seasonality_type)
    adjusted_forecast = []
    
    for i, value in enumerate(forecast_values):
        month = ((start_month - 1 + i) % 12) + 1  # Convert to 1-12 range
        multiplier = seasonal_multipliers.get(month, 1.0)
        adjusted_forecast.append(value * multiplier)
    
    return np.array(adjusted_forecast)

def get_seasonal_multipliers(seasonality_type):
    """Get seasonal adjustment multipliers by month"""
    if seasonality_type == "holiday":
        return {
            1: 0.85, 2: 0.80, 3: 0.90, 4: 0.95, 5: 0.95, 6: 0.95,
            7: 0.90, 8: 0.90, 9: 1.00, 10: 1.15, 11: 1.30, 12: 1.35
        }
    elif seasonality_type == "anti_holiday":
        return {
            1: 1.10, 2: 1.15, 3: 1.05, 4: 1.00, 5: 1.00, 6: 1.00,
            7: 1.05, 8: 1.05, 9: 1.00, 10: 0.90, 11: 0.85, 12: 0.80
        }
    elif seasonality_type == "summer":
        return {
            1: 0.90, 2: 0.90, 3: 0.95, 4: 1.00, 5: 1.05, 6: 1.20,
            7: 1.25, 8: 1.15, 9: 1.00, 10: 0.95, 11: 0.90, 12: 0.95
        }
    else:
        return {i: 1.0 for i in range(1, 13)}

def calculate_forecast_confidence(historical_values, model_type):
    """Calculate confidence intervals for forecasts"""
    if len(historical_values) < 3:
        return 0.3  # High uncertainty for limited data
    
    # Calculate historical forecast errors (simplified)
    errors = []
    for i in range(3, len(historical_values)):
        actual = historical_values[i]
        simple_forecast = np.mean(historical_values[i-3:i])
        error = abs(actual - simple_forecast) / (actual + 1)
        errors.append(error)
    
    if not errors:
        base_uncertainty = 0.2
    else:
        base_uncertainty = np.mean(errors)
    
    # Adjust uncertainty by model type
    model_adjustments = {
        "Prophet_Enhanced": 0.8,  # Lower uncertainty
        "ARIMA_Enhanced": 0.9,
        "Exponential_Smoothing_Enhanced": 1.0,
        "Simple_Trend": 1.2  # Higher uncertainty
    }
    
    adjustment = model_adjustments.get(model_type, 1.0)
    return min(base_uncertainty * adjustment, 0.5)  # Cap at 50%

def validate_forecast_quality(historical_values, forecast_values, characteristics):
    """Validate forecast makes business sense and handle discontinued products"""
    if len(historical_values) == 0:
        return np.maximum(forecast_values, 0), "No historical data for validation"
    
    # Handle discontinued products - allow zero forecasts
    if characteristics["stage"] == "discontinued":
        return np.zeros_like(forecast_values), "Discontinued product - zero forecast"
    
    recent_avg = np.mean(historical_values[-3:]) if len(historical_values) >= 3 else np.mean(historical_values)
    forecast_avg = np.mean(forecast_values[:6])  # First 6 months
    
    warnings = []
    
    # CRITICAL: Ensure no negative values at any point
    forecast_values = np.maximum(forecast_values, 0)
    
    # Check for unrealistic growth
    growth_ratio = forecast_avg / (recent_avg + 0.1)
    if growth_ratio > 2.0:
        warnings.append("High growth forecast")
        # Cap extreme growth
        forecast_values = forecast_values * min(2.0 / growth_ratio, 1.0)
    
    # Check for unrealistic decline (but don't allow negatives)
    elif growth_ratio < 0.1:  # Changed from 0.3 to 0.1 for more conservative floor
        warnings.append("Steep decline forecast")
        # Floor extreme decline but ensure minimum of 0
        min_floor = max(0.1 * recent_avg, 1)  # At least 10% of recent average or 1 unit
        forecast_values = np.maximum(forecast_values, min_floor)
    
    # For active products, ensure minimum viable sales
    if characteristics["stage"] not in ["discontinued", "decline"]:
        min_viable = max(1, recent_avg * 0.1)  # At least 10% of recent average or 1 unit minimum
        forecast_values = np.maximum(forecast_values, min_viable)
    elif characteristics["stage"] == "decline":
        # Allow declining products to go to zero over time, but not negative
        forecast_values = np.maximum(forecast_values, 0)
    
    # Final safety check: absolutely no negatives allowed
    forecast_values = np.maximum(forecast_values, 0)
    
    return forecast_values, "; ".join(warnings) if warnings else "Passed validation"

def add_growth_trend(forecast_values, growth_rate=0.05):
    """Add a customizable growth trend to forecasts with negative value protection"""
    trend_multipliers = [(1 + growth_rate) ** (i/12) for i in range(len(forecast_values))]
    result = forecast_values * np.array(trend_multipliers)
    
    # Ensure no negative values regardless of growth rate
    return np.maximum(result, 0)

def create_simple_forecast(ts, steps, growth_rate):
    """Simple forecasting method for very short time series with negative value protection"""
    try:
        values = ts.values
        if len(values) < 2:
            # Use last value with growth
            base_value = max(values[-1], 1) if len(values) > 0 else 1  # Ensure positive base
        else:
            # Simple trend extrapolation
            if len(values) >= 3:
                trend = np.polyfit(range(len(values)), values, 1)[0]
                base_value = max(values[-1] + trend, 1)  # Ensure positive base
            else:
                base_value = max(np.mean(values), 1)  # Ensure positive base
        
        # Apply growth trend
        forecast = []
        for i in range(steps):
            growth_multiplier = (1 + growth_rate) ** (i/12)
            forecast_val = base_value * growth_multiplier
            forecast.append(max(0, forecast_val))  # Ensure non-negative
        
        return np.array(forecast)
    
    except:
        return None

def fit_arima_model(ts, steps=18, growth_rate=0.05):
    """Fit ARIMA model optimized for trend-focused products"""
    try:
        # More focused parameter search for stability
        p_values = range(0, 2)  # Reduced range
        d_values = range(0, 2)  
        q_values = range(0, 2)
        
        best_aic = float('inf')
        best_model = None
        
        for order in itertools.product(p_values, d_values, q_values):
            try:
                model = ARIMA(ts, order=order)
                fitted_model = model.fit()
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_model = fitted_model
            except:
                continue
        
        if best_model is None:
            return None, float('inf'), "ARIMA_Failed"
        
        # Generate forecast
        forecast = best_model.forecast(steps=steps)
        
        # Apply growth trend
        forecast = add_growth_trend(forecast, growth_rate=growth_rate)
        forecast = np.maximum(forecast, 0)
        
        return forecast, best_aic, "ARIMA_Enhanced"
        
    except Exception as e:
        return None, float('inf'), f"ARIMA_Error: {str(e)}"

def fit_exponential_smoothing(ts, steps=18, growth_rate=0.05):
    """Fit Exponential Smoothing optimized for stable/mature products"""
    try:
        # Simplified configurations for stability
        configs = [
            {'trend': 'add', 'damped_trend': True},
            {'trend': 'add', 'damped_trend': False},
            {'trend': None}
        ]
        
        best_aic = float('inf')
        best_forecast = None
        
        for config in configs:
            try:
                model = ExponentialSmoothing(ts, seasonal=None, **config)
                fitted_model = model.fit(optimized=True, use_brute=False)
                forecast = fitted_model.forecast(steps=steps)
                
                # Apply growth trend
                forecast = add_growth_trend(forecast, growth_rate=growth_rate)
                forecast = np.maximum(forecast, 0)
                
                if fitted_model.aic < best_aic:
                    best_aic = fitted_model.aic
                    best_forecast = forecast
            except:
                continue
        
        if best_forecast is not None:
            return best_forecast, best_aic, "Exponential_Smoothing_Enhanced"
        else:
            return None, float('inf'), "Exponential_Smoothing_Error"
            
    except Exception as e:
        return None, float('inf'), f"Exponential_Smoothing_Error: {str(e)}"

def fit_prophet_model(df_asin, steps=18, growth_rate=0.05):
    """Fit Prophet model optimized for seasonal products"""
    if not PROPHET_AVAILABLE:
        return None, float('inf'), "Prophet_Not_Available"
    
    try:
        # Prepare data for Prophet
        prophet_df = df_asin[['Date', 'Units_Sold']].rename(columns={'Date': 'ds', 'Units_Sold': 'y'})
        
        if len(prophet_df) < 8:
            return None, float('inf'), "Prophet_Insufficient_Data"
        
        # Conservative parameters for short series
        model = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.001,
            seasonality_prior_scale=0.01,
            growth='linear'
        )
        
        # Fit the model
        model.fit(prophet_df)
        
        # Create future dataframe
        last_date = prophet_df['ds'].max()
        future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), 
            periods=steps, 
            freq='MS'
        )
        future_df = pd.DataFrame({'ds': future_dates})
        
        # Make forecast
        forecast = model.predict(future_df)
        forecasted_values = forecast['yhat'].values
        
        # Apply growth trend
        forecasted_values = add_growth_trend(forecasted_values, growth_rate=growth_rate)
        forecasted_values = np.maximum(forecasted_values, 0)
        
        # Calculate pseudo-AIC
        in_sample_forecast = model.predict(prophet_df)
        mse = np.mean((prophet_df['y'] - in_sample_forecast['yhat']) ** 2)
        pseudo_aic = len(prophet_df) * np.log(mse + 1e-8) + 2 * 3
        
        return forecasted_values, pseudo_aic, "Prophet_Enhanced"
        
    except Exception as e:
        return None, float('inf'), f"Prophet_Error: {str(e)}"

def forecast_asin(df_asin, asin, steps=18, user_growth_rate=0.05):
    """Forecast for a single ASIN using intelligent model selection and business logic"""
    try:
        # Prepare time series
        ts = df_asin.set_index('Date')['Units_Sold']
        
        # Ensure we have enough data points
        if len(ts) < 3:
            return None, f"Insufficient data (only {len(ts)} points)"
        
        # Analyze product characteristics
        characteristics = analyze_product_characteristics(ts)
        
        # Use product-specific growth rate, but allow user override for mature products
        if characteristics["stage"] == "mature":
            growth_rate = user_growth_rate  # Use user's business judgment for mature products
        else:
            growth_rate = characteristics["growth_rate"]  # Use data-driven rates for others
        
        # Select best model based on characteristics
        models_results = []
        
        # Smart model selection based on data characteristics
        if len(ts) < 6:
            # For very short series, use simple methods
            simple_forecast = create_simple_forecast(ts, steps, growth_rate)
            if simple_forecast is not None:
                models_results.append((simple_forecast, 100, "Simple_Trend"))
        
        else:
            # ARIMA - better for trend-focused products
            if characteristics["stage"] in ["growth", "decline"]:
                arima_forecast, arima_aic, arima_name = fit_arima_model(ts, steps, growth_rate)
                if arima_forecast is not None:
                    models_results.append((arima_forecast, arima_aic, arima_name))
            
            # Exponential Smoothing - good for stable/mature products
            if characteristics["stage"] in ["mature", "early_mature"]:
                es_forecast, es_aic, es_name = fit_exponential_smoothing(ts, steps, growth_rate)
                if es_forecast is not None:
                    models_results.append((es_forecast, es_aic, es_name))
            
            # Prophet - best for seasonal products with enough data
            if len(ts) >= 8 and PROPHET_AVAILABLE and characteristics["seasonality"] != "none":
                prophet_forecast, prophet_aic, prophet_name = fit_prophet_model(df_asin, steps, growth_rate)
                if prophet_forecast is not None:
                    models_results.append((prophet_forecast, prophet_aic, prophet_name))
            
            # Always try Exponential Smoothing as fallback
            if len(models_results) == 0:
                es_forecast, es_aic, es_name = fit_exponential_smoothing(ts, steps, growth_rate)
                if es_forecast is not None:
                    models_results.append((es_forecast, es_aic, es_name))
        
        # Select best model
        if not models_results:
            return None, "All models failed"
        
        best_forecast, best_aic, best_model = min(models_results, key=lambda x: x[1])
        
        # Apply seasonal adjustments
        last_date = df_asin['Date'].max()
        start_month = (last_date + pd.DateOffset(months=1)).month
        best_forecast = apply_seasonal_adjustments(best_forecast, characteristics["seasonality"], start_month)
        
        # Ensure no negative values after seasonal adjustments
        best_forecast = np.maximum(best_forecast, 0)
        
        # Business logic validation and adjustment
        best_forecast, validation_msg = validate_forecast_quality(ts.values, best_forecast, characteristics)
        
        # Final safety check for negative values
        best_forecast = np.maximum(best_forecast, 0)
        
        # Calculate confidence intervals
        confidence = calculate_forecast_confidence(ts.values, best_model)
        
        # Round forecasts to whole numbers (ensure still non-negative)
        best_forecast = np.maximum(np.round(best_forecast).astype(int), 0)
        
        # Create forecast dates
        forecast_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                     periods=steps, freq='MS')
        
        # Enhanced model name with characteristics
        enhanced_model_name = f"{best_model}_{characteristics['stage']}_{characteristics['seasonality']}"
        
        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Date': forecast_dates,
            'ASIN': asin,
            'Units_Sold': best_forecast,
            'Type': 'Forecast',
            'Model_Used': enhanced_model_name,
            'Product_Stage': characteristics['stage'],
            'Seasonality': characteristics['seasonality'],
            'Growth_Rate': f"{growth_rate:.1%}",
            'Confidence': f"{confidence:.1%}",
            'Validation': validation_msg
        })
        
        return forecast_df, None
        
    except Exception as e:
        return None, f"Error: {str(e)}"

def create_enhanced_plot(df, asin_sample, combined_df):
    """Create an enhanced plot with additional business context"""
    asin_data = combined_df[combined_df['ASIN'] == asin_sample].copy()
    
    fig = go.Figure()
    
    # Historical data
    historical = asin_data[asin_data['Type'] == 'Actual']
    fig.add_trace(go.Scatter(
        x=historical['Date'],
        y=historical['Units_Sold'],
        mode='lines+markers',
        name='Historical Sales',
        line=dict(color='blue', width=3),
        marker=dict(size=6)
    ))
    
    # Forecasted data
    forecasted = asin_data[asin_data['Type'] == 'Forecast']
    if not forecasted.empty:
        fig.add_trace(go.Scatter(
            x=forecasted['Date'],
            y=forecasted['Units_Sold'],
            mode='lines+markers',
            name='AI Forecast',
            line=dict(color='red', dash='dash', width=3),
            marker=dict(size=6)
        ))
        
        # Add confidence bands if available
        if 'Confidence' in forecasted.columns:
            confidence_pct = float(forecasted['Confidence'].iloc[0].strip('%')) / 100
            upper_bound = forecasted['Units_Sold'] * (1 + confidence_pct)
            lower_bound = forecasted['Units_Sold'] * (1 - confidence_pct)
            
            fig.add_trace(go.Scatter(
                x=forecasted['Date'],
                y=upper_bound,
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            fig.add_trace(go.Scatter(
                x=forecasted['Date'],
                y=lower_bound,
                fill='tonexty',
                mode='lines',
                line=dict(width=0),
                name='Confidence Interval',
                fillcolor='rgba(255,0,0,0.2)',
                hoverinfo='skip'
            ))
    
    # Enhanced title with characteristics
    if not forecasted.empty:
        stage = forecasted['Product_Stage'].iloc[0].title()
        seasonality = forecasted['Seasonality'].iloc[0].title()
        growth = forecasted['Growth_Rate'].iloc[0]
        title = f'Enhanced Forecast: {asin_sample}<br><sub>Stage: {stage} | Seasonality: {seasonality} | Growth: {growth}</sub>'
    else:
        title = f'Sales History: {asin_sample}'
    
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Units Sold',
        hovermode='x unified',
        showlegend=True,
        height=500
    )
    
    return fig

def main():
    st.set_page_config(page_title="ASIN Time Series Forecasting", layout="wide")
    
    st.title("🔮 ASIN Time Series Forecasting App")
    st.markdown("Upload your monthly unit sales data CSV to get 18-month forecasts for each ASIN using automated model selection!")
    
    # Sidebar for instructions and controls
    with st.sidebar:
        st.header("📋 Quick Setup")
        st.markdown("""
        **CSV Requirements:**
        - Columns: `Date`, `ASIN`, `Units_Sold` (exact names)
        - Excel Short Date format is supported
        - Missing months = 0 sales (auto-filled)
        """)
        
        # Template download
        template_data = "Date,ASIN,Units_Sold\n2023-01-01,B001,100\n2023-02-01,B001,120\n2023-03-01,B001,95"
        st.download_button(
            label="📄 Download CSV Template",
            data=template_data,
            file_name="asin_forecast_template.csv",
            mime="text/csv"
        )
        
        st.markdown("---")
        
        # Growth rate input
        st.header("📈 Growth Settings")
        user_growth_rate = st.slider(
            "Expected Annual Growth Rate (%)",
            min_value=-20.0,
            max_value=50.0,
            value=5.0,
            step=0.5,
            help="Applied to mature products. Growing/declining products use data-driven rates."
        ) / 100
        
        st.markdown("---")
        st.markdown("""
        **Enhanced AI Features:**
        - Automatic product lifecycle detection
        - Intelligent seasonality recognition  
        - Smart model selection by product type
        - Business logic validation
        - Confidence interval estimation
        
        **Model Priority:**
        - Growth products → ARIMA (trend-focused)
        - Seasonal products → Prophet (pattern-aware)
        - Mature products → Exponential Smoothing (stable)
        """)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    # Initialize session state for results persistence
    if 'forecast_results' not in st.session_state:
        st.session_state.forecast_results = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    
    if uploaded_file is not None:
        # Check if this is a new file
        file_id = uploaded_file.name + str(uploaded_file.size)
        if st.session_state.current_file != file_id:
            st.session_state.current_file = file_id
            st.session_state.forecast_results = None
        
        # Load and validate data
        df = load_and_validate_data(uploaded_file)
        
        if df is not None:
            # Display data summary
            st.subheader("📊 Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total ASINs", df['ASIN'].nunique())
            with col2:
                st.metric("Total Records", len(df))
            with col3:
                date_range = (df['Date'].max() - df['Date'].min()).days // 30
                st.metric("Months of History", date_range)
            with col4:
                st.metric("Date Range", f"{df['Date'].min().strftime('%Y-%m')} to {df['Date'].max().strftime('%Y-%m')}")
            
            # Show sample data
            st.subheader("📋 Sample Data")
            st.dataframe(df.head(10))
            
            # Auto-start forecasting if no results exist
            if st.session_state.forecast_results is None:
                asins = df['ASIN'].unique()
                
                # Fun loading messages with improved styling
                loading_messages = [
                    "I asked my media buyer to optimize the campaign. She lit a candle and whispered 'ROAS' three times.",
                    "My conversion rate's like my gym membership—low effort, high expectations.",
                    "Sponsored Products are like reality stars. Flashy, visible, and nobody knows if they actually work.",
                    "Amazon said we were exclusive. Then I saw them with 6 other brands on the same shelf.",
                    "My PDP ghosted me. Turns out it just wasn't converting anymore.",
                    "Every time someone says 'upper funnel,' a KPI dies inside.",
                    "You know you work in e-comm when you say 'PDP' more than you say 'I love you.'",
                    "Target's red dot isn't a logo. It's a bullseye on your wallet.",
                    "If this bid strategy had a heartbeat, we'd be legally obligated to pull the plug.",
                    "At this point, I don't know if I'm optimizing the campaign or the campaign is gaslighting me.",
                    "I asked Alexa how to boost my ROAS. She played 'Fix You' by Coldplay.",
                    "Our Display ads are like office coffee—technically there, but nobody remembers engaging with them.",
                    "The ad isn't underperforming, the customer is.",
                    "Dating an Amazon algorithm is wild. One day you're the #1 choice. The next, 'customers also bought literally anything else.'",
                    "I don't always fall in love, but when I do, it's with a cart that hasn't been abandoned.",
                    "Programmatic isn't confusing. It's just 4,000 acronyms pretending to know you better than your spouse.",
                    "I clicked on my own ad just to feel something.",
                    "Walmart's pricing strategy is like a suspense movie. Just when you think it can't go lower… it rolls back again."
                ]
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                fun_message_container = st.empty()
                
                combined_results = []
                model_summary = {}
                failed_asins = []
                
                import random
                import time
                
                # Shuffle the loading messages for better randomization each run
                random.shuffle(loading_messages)
                
                # Show loading messages with enhanced styling
                colors = ['#1E3A8A', '#7C2D12', '#14532D', '#4C1D95', '#92400E', '#701A75', '#0F766E', '#B91C1C']
                message_index = 0
                last_message_time = time.time()
                
                for i, asin in enumerate(asins):
                    # Show fun message every 7 seconds or at start
                    current_time = time.time()
                    if (current_time - last_message_time >= 7 or i == 0) and message_index < len(loading_messages):
                        color = random.choice(colors)
                        with fun_message_container.container():
                            st.markdown(f"""
                            <div style="background-color: #f8fafc; padding: 20px; border-radius: 15px; border-left: 6px solid {color}; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px 0;">
                                <p style="color: {color}; font-size: 22px; font-style: italic; margin: 0; font-weight: 600; line-height: 1.4;">
                                    💭 {loading_messages[message_index]}
                                </p>
                            </div>
                            """, unsafe_allow_html=True)
                        message_index += 1
                        last_message_time = current_time
                        
                        # If we've shown all messages, reshuffle for next batch
                        if message_index >= len(loading_messages):
                            random.shuffle(loading_messages)
                            message_index = 0
                    
                    status_text.text(f'Processing ASIN {i+1}/{len(asins)}: {asin}')
                    
                    # Get data for this ASIN
                    asin_data = df[df['ASIN'] == asin].copy()
                    
                    # Add historical data to results
                    historical_data = asin_data.copy()
                    historical_data['Type'] = 'Actual'
                    historical_data['Model_Used'] = 'N/A'
                    combined_results.append(historical_data)
                    
                    # Generate forecast
                    forecast_df, error = forecast_asin(asin_data, asin, user_growth_rate=user_growth_rate)
                    
                    if forecast_df is not None:
                        combined_results.append(forecast_df)
                        model_summary[asin] = forecast_df['Model_Used'].iloc[0]
                    else:
                        failed_asins.append((asin, error))
                        model_summary[asin] = f"Failed: {error}"
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(asins))
                
                fun_message_container.empty()
                status_text.text('✅ Forecasting completed!')
                
                # Store results in session state
                if combined_results:
                    final_df = pd.concat(combined_results, ignore_index=True)
                    final_df = final_df.sort_values(['ASIN', 'Date'])
                    st.session_state.forecast_results = {
                        'final_df': final_df,
                        'model_summary': model_summary,
                        'failed_asins': failed_asins,
                        'asins': asins
                    }
            
            # Display results if they exist
            if st.session_state.forecast_results is not None:
                results = st.session_state.forecast_results
                final_df = results['final_df']
                model_summary = results['model_summary']
                failed_asins = results['failed_asins']
                asins = results['asins']
                
                st.success(f"Forecasting completed! Generated forecasts for {len(asins) - len(failed_asins)} out of {len(asins)} ASINs using intelligent model selection.")
                
                # Enhanced model summary with product stages
                st.subheader("🤖 Intelligent Model Selection Results")
                
                # Show product stage distribution
                if 'Product_Stage' in final_df.columns:
                    forecast_data = final_df[final_df['Type'] == 'Forecast']
                    if not forecast_data.empty:
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            stage_counts = forecast_data.groupby('ASIN')['Product_Stage'].first().value_counts()
                            st.write("**Product Lifecycle Stages:**")
                            for stage, count in stage_counts.items():
                                st.write(f"• {stage.title()}: {count} ASINs")
                        
                        with col2:
                            seasonality_counts = forecast_data.groupby('ASIN')['Seasonality'].first().value_counts()
                            st.write("**Seasonality Patterns:**")
                            for pattern, count in seasonality_counts.items():
                                st.write(f"• {pattern.title()}: {count} ASINs")
                        
                        with col3:
                            model_counts = forecast_data.groupby('ASIN')['Model_Used'].first().value_counts()
                            st.write("**Models Selected:**")
                            for model, count in model_counts.items():
                                model_name = model.split('_')[0] if '_' in model else model
                                st.write(f"• {model_name}: {count} ASINs")
                
                # Enhanced visualization
                st.subheader("📈 Enhanced Forecast Visualization")
                successful_asins = [asin for asin in asins if asin not in [f[0] for f in failed_asins]]
                if successful_asins:
                    # Sort ASINs by total units sold (descending)
                    asin_totals = df.groupby('ASIN')['Units_Sold'].sum().sort_values(ascending=False)
                    sorted_successful_asins = [asin for asin in asin_totals.index if asin in successful_asins]
                    
                    sample_asin = st.selectbox("Select ASIN to visualize (sorted by total sales):", sorted_successful_asins, key="asin_selector")
                    
                    # Show ASIN characteristics
                    asin_forecast = final_df[(final_df['ASIN'] == sample_asin) & (final_df['Type'] == 'Forecast')]
                    if not asin_forecast.empty:
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Product Stage", asin_forecast['Product_Stage'].iloc[0].title())
                        with col2:
                            st.metric("Seasonality", asin_forecast['Seasonality'].iloc[0].title())
                        with col3:
                            st.metric("Growth Rate", asin_forecast['Growth_Rate'].iloc[0])
                        with col4:
                            st.metric("Confidence", asin_forecast['Confidence'].iloc[0])
                    
                    fig = create_enhanced_plot(df, sample_asin, final_df)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Show failed ASINs if any
                if failed_asins:
                    st.subheader("⚠️ Failed ASINs")
                    failed_df = pd.DataFrame(failed_asins, columns=['ASIN', 'Error'])
                    st.dataframe(failed_df)
                
                # Enhanced download section
                st.subheader("📥 Download Enhanced Results")
                
                # Convert to CSV
                csv_data = final_df.to_csv(index=False)
                
                col1, col2 = st.columns([1, 1])
                with col1:
                    st.download_button(
                        label="📁 Download Complete Dataset",
                        data=csv_data,
                        file_name=f"enhanced_asin_forecasts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_button"
                    )
                
                with col2:
                    # Show enhanced dataset preview
                    forecast_preview = final_df[final_df['Type'] == 'Forecast'].head()
                    st.text(f"Enhanced dataset with {len(final_df)} total records")
                    if not forecast_preview.empty:
                        st.dataframe(forecast_preview[['Date', 'ASIN', 'Units_Sold', 'Product_Stage', 'Seasonality', 'Growth_Rate']])
    
    else:
        st.info("👆 Please upload a CSV file to get started!")
        
        # Show sample data format
        st.subheader("📋 Sample Data Format")
        sample_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-02-01', '2023-03-01', '2023-01-01', '2023-02-01'],
            'ASIN': ['B001', 'B001', 'B001', 'B002', 'B002'],
            'Units_Sold': [100, 120, 95, 250, 280]
        })
        st.dataframe(sample_data)

if __name__ == "__main__":
    main()
