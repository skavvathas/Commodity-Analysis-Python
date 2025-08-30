# Commodity Analysis in Python

This project provides a comprehensive **quantitative analysis of commodity ETFs**, focusing on oil, gold, agriculture, and silver. Using Python and historical market data, the study explores **volatility, correlations, risk metrics, and portfolio diversification** strategies to better understand commodity market behavior.

## Project Overview

The analysis covers several key areas:

1. **Data Collection & Preprocessing**
   - Downloaded historical price data for ETFs: **USO (Oil), GLD (Gold), DBA (Agriculture), SLV (Silver)**.
   - Calculated daily percentage returns and cleaned the data by removing missing values.

2. **Volatility Analysis**
   - Computed **20-day rolling annualized volatility** to visualize risk trends over time.
   - Compared volatility patterns across commodities, highlighting oil’s high variability versus gold’s more stable behavior.
   - Identified periods of extreme volatility linked to major market events such as the **2014-2016 Oil Crash**, **COVID-19 pandemic**, and recent geopolitical shocks.

3. **Return Distribution & Statistics**
   - Visualized **histograms and KDE plots** of daily returns to understand distribution characteristics.
   - Calculated descriptive statistics including **mean, standard deviation, skewness, and kurtosis** for each commodity.

4. **Risk Metrics**
   - Estimated **Value at Risk (VaR)** at a 95% confidence level to quantify potential losses under normal market conditions.
   - Analyzed the **correlation matrix** of returns to understand co-movements between commodities.
   - Created heatmaps for intuitive visualization of relationships between assets.

5. **Hedging Analysis**
   - Applied **OLS regression** to estimate a hedge ratio for using gold (GLD) to hedge oil (USO) exposure.
   - Constructed a **hedged return series** and demonstrated **volatility reduction** achieved through hedging strategies.

6. **Portfolio Construction & Diversification**
   - Built a **simple two-commodity portfolio** (oil and gold) and later expanded to a **four-commodity portfolio** (oil, gold, agriculture, silver).
   - Calculated expected **annualized portfolio returns and volatility**, highlighting the benefits of diversification across different commodities.

## Key Insights

- Oil (USO) is **highly volatile**, while gold (GLD) tends to be **more stable**, making it a potential hedge in commodity portfolios.
- Multi-commodity portfolios reduce overall portfolio volatility and enhance risk-adjusted performance.
- Hedge ratios provide actionable strategies for mitigating downside risk in volatile markets.
- Correlation analysis illustrates which commodities move together and which provide diversification benefits.

## Tools & Libraries

- **Python**: Data processing and analysis
- **yfinance**: Historical price data retrieval
- **pandas / numpy**: Data manipulation and calculations
- **matplotlib / seaborn**: Visualization
- **scipy.stats**: Risk metrics and statistical calculations
- **statsmodels**: Regression and hedging analysis
