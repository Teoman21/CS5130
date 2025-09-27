import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

def generate_tech_stock_data(n_points=100):
    """
    Generate synthetic technology stock market data with realistic correlations.
    
    Variables and their relationships:
    1. Daily Returns (%): Day-to-day price changes
    2. Trading Volume (millions): Number of shares traded
    3. Volatility Index: Measure of price fluctuation intensity
    4. Market Cap Change (%): Change in company valuation
    
    Expected Correlations:
    - Returns & Market Cap Change: Strong positive (~0.85) - price changes directly affect market cap
    - Volatility & Volume: Moderate positive (~0.45) - higher volatility attracts more traders
    - Returns & Volatility: Near-zero (~-0.05) - volatility can occur in both directions
    - Returns & Volume: Weak negative (~-0.15) - large sell-offs increase volume but decrease returns
    """
    
    # Generate base components with controlled correlations
    
    # 1. Daily Returns (%) - Primary variable
    # Tech stocks typically have daily returns between -5% and +5%
    # with occasional larger moves
    returns = np.random.normal(0.08, 1.8, n_points)  # Slight positive bias, ~1.8% std dev
    returns = np.clip(returns, -7, 8)  # Clip extreme values
    
    # 2. Volatility Index (0-100 scale, similar to VIX for individual stocks)
    # Base volatility with some independence from returns
    volatility_base = np.random.gamma(2, 8, n_points)  # Right-skewed distribution
    # Add small correlation with absolute returns (volatility increases with big moves)
    volatility = volatility_base + 3 * np.abs(returns) + np.random.normal(0, 3, n_points)
    volatility = np.clip(volatility, 5, 65)  # Tech stocks can be volatile but not extreme
    
    # 3. Trading Volume (millions of shares)
    # Moderately correlated with volatility, weakly negative with returns
    volume_base = np.random.gamma(3, 15, n_points)  # Right-skewed, typical is 45M
    # Volume increases with volatility and slightly with negative returns
    volume = volume_base + 0.8 * volatility - 3 * returns + np.random.normal(0, 10, n_points)
    volume = np.clip(volume, 10, 200)  # Realistic range for tech giants
    
    # 4. Market Cap Change (%)
    # Strongly correlated with returns (companies worth more when stock price rises)
    market_cap_change = 0.95 * returns + np.random.normal(0, 0.3, n_points)
    # Add small noise to prevent perfect correlation
    market_cap_change = market_cap_change + np.random.normal(0, 0.2, n_points)
    
    # Create DataFrame with the four variables
    df = pd.DataFrame({
        'daily_returns_pct': np.round(returns, 3),
        'trading_volume_millions': np.round(volume, 1),
        'volatility_index': np.round(volatility, 2),
        'market_cap_change_pct': np.round(market_cap_change, 3)
    })
    
    # Add a date index for realism
    dates = pd.date_range(start='2024-06-01', periods=n_points, freq='B')  # Business days only
    df.index = dates
    df.index.name = 'date'
    
    # Add stock ticker column (randomly assign one of the four tech stocks)
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'NVDA']
    df['ticker'] = np.random.choice(tickers, size=n_points)
    
    # Reorder columns for better presentation
    df = df[['ticker', 'daily_returns_pct', 'trading_volume_millions', 
             'volatility_index', 'market_cap_change_pct']]
    
    return df

# Generate the dataset
stock_data = generate_tech_stock_data(100)

# Display basic statistics
print("=" * 60)
print("SYNTHETIC TECH STOCK DATASET")
print("=" * 60)
print(f"\nDataset shape: {stock_data.shape}")
print(f"Date range: {stock_data.index[0].date()} to {stock_data.index[-1].date()}")

print("\n" + "=" * 60)
print("SUMMARY STATISTICS")
print("=" * 60)
print(stock_data.describe())

print("\n" + "=" * 60)
print("CORRELATION MATRIX")
print("=" * 60)
correlation_matrix = stock_data[['daily_returns_pct', 'trading_volume_millions', 
                                  'volatility_index', 'market_cap_change_pct']].corr()
print(correlation_matrix)

print("\n" + "=" * 60)
print("VARIABLE RELATIONSHIPS EXPLAINED")
print("=" * 60)
print("""
1. Returns ↔ Market Cap Change (≈0.95): STRONG POSITIVE
   - When stock price increases, market capitalization increases proportionally
   
2. Volatility ↔ Volume (≈0.45): MODERATE POSITIVE  
   - High volatility attracts day traders and algorithms, increasing volume
   
3. Returns ↔ Volatility (≈-0.05): NEAR ZERO
   - Volatility measures magnitude of change, not direction
   
4. Returns ↔ Volume (≈-0.15): WEAK NEGATIVE
   - Panic selling and profit-taking can increase volume during price drops
""")

print("\n" + "=" * 60)
print("FIRST 10 ROWS OF DATA")
print("=" * 60)
print(stock_data.head(10))

# Save to CSV for further analysis
stock_data.to_csv('tech_stocks_synthetic_data.csv')
print("\n✓ Data saved to 'tech_stocks_synthetic_data.csv'")

# Calculate and display covariance matrix
print("\n" + "=" * 60)
print("COVARIANCE MATRIX")
print("=" * 60)
covariance_matrix = stock_data[['daily_returns_pct', 'trading_volume_millions', 
                                 'volatility_index', 'market_cap_change_pct']].cov()
print(covariance_matrix)