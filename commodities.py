#!/usr/bin/env python
# coding: utf-8

# # Commodity Volatility Analysis in Python

# In[1]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from scipy.stats import norm


# ### Download historical price data for USO and GLD ETFs

# ### USO (United States Oil Fund)
# 
# - This ETF tracks the price of West Texas Intermediate (WTI) crude oil
# - It's designed to reflect the daily changes in percentage terms of the price of WTI crude oil
# - USO holds oil futures contracts and rolls them forward to maintain exposure
# - It's one of the most popular ways for retail investors to gain exposure to oil prices without trading futures directly
# 
# ### GLD (SPDR Gold Shares)
# 
# - This ETF tracks the price of gold bullion
# - It's backed by physical gold stored in vaults, primarily in London
# - Each share represents a fractional ownership of gold (approximately 1/10th of an ounce)
# - GLD is the largest and most liquid gold ETF, making it a standard choice for gold exposure

# In[2]:


symbols = ['USO', 'GLD']
data = yf.download(symbols, start='2010-01-01', end='2025-06-30')
data


# In[3]:


data['Close']


# Remove any rows that contain missing values 

# In[4]:


data.dropna(inplace=True)
data


# Calculate the daily percentage returns for your commodities and remove the first row which will be NaN.

# In[5]:


returns = data.pct_change().dropna()
returns


# # Rolling volatility (20-day window)
# 
# Calculate the annualized rolling volatility for your commodities, which measures how much the prices fluctuate over time

# ## What is Annualized Rolling Volatility?
# Annualized rolling volatility is a measure that tells you how much an asset's price is expected to fluctuate over a full year, based on recent daily price movements. It's expressed as a percentage and updates daily as new data comes in.
# Think of it as a "risk thermometer" that shows:
# 
# - Low volatility (10-20%): Relatively stable, predictable price movements
# - Medium volatility (20-40%): Moderate price swings
# - High volatility (40%+): Wild, unpredictable price movements

# In[6]:


volatility = returns.rolling(window=20).std() * np.sqrt(252)
volatility


# In[7]:


volatility.dropna()
volatility


# ## Line chart comparing the volatility patterns of GLD and USO over time

# In[8]:


vol_close = volatility['Close']  # Now this is a regular DataFrame: columns = ['GLD', 'USO']

plt.figure(figsize=(14, 7))
for symbol in symbols:
    plt.plot(vol_close[symbol], label=f'{symbol} Volatility')
plt.title('Rolling Annualized Volatility (20-day window)')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()


# What we see in the resulting chart:
# 
# - USO line: Likely more jagged with higher peaks (oil is volatile)
# - GLD line: Probably smoother and lower (gold is more stable)
# 
# Volatility spikes: During major events like:
# 
# - 2014-2016 Oil Crash
# - 2020 COVID Pandemic
# - Recent geopolitical events

# ### Histogram of returns

# In[9]:


returns_close = returns['Close']

plt.figure(figsize=(10, 5))
for symbol in symbols:
    sns.histplot(returns_close[symbol], kde=True, stat="density", label=symbol, element="step")
plt.title('Return Distributions')
plt.legend()
plt.show()


# ### Descriptive stats

# In[10]:


print(returns_close.describe())
print("\nSkewness:\n", returns_close.skew())
print("\nKurtosis:\n", returns_close.kurtosis())


# ## Value at Risk (VaR) Analysis - 95% Confidence Level

# In[11]:


confidence_level = 0.95
VaR = {}

for symbol in symbols:
    mean_return = returns_close[symbol].mean()
    std_dev = returns_close[symbol].std()
    VaR[symbol] = norm.ppf(1 - confidence_level, mean_return, std_dev)

print(f"Value at Risk (95% confidence): {VaR}")


# ## Correlation of daily returns

# In[12]:


correlation_matrix = returns_close.corr()
correlation_matrix


# ## Heatmap

# In[13]:


plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Commodity Returns')
plt.show()


# ## A simple hedge ratio can be estimated using OLS regression.

# Perform a hedge ratio analysis to determine how to use gold (GLD) to hedge against oil (USO) price movements.

# In[14]:


import statsmodels.api as sm

# Hedge example: Hedge Oil (USO) using Gold (GLD)
X = returns_close['GLD']
Y = returns_close['USO']
X = sm.add_constant(X)

model = sm.OLS(Y, X).fit()
print(model.summary())

hedge_ratio = model.params['GLD']
print(f"\nEstimated Hedge Ratio: {hedge_ratio:.4f}")


# ## Construct hedged return series

# In[15]:


hedged_return = returns_close['USO'] - hedge_ratio * returns_close['GLD']
hedged_return


# ## Compare standard deviation (volatility)

# In[16]:


original_vol = returns_close['USO'].std()
hedged_vol = hedged_return.std()

print(f"\nOriginal Volatility (USO): {original_vol:.4f}")
print(f"Hedged Volatility: {hedged_vol:.4f}")
print(f"Volatility Reduction: {(1 - hedged_vol/original_vol) * 100:.2f}%")


# ## Lets build a simple 2-commodities portfolio to analyze diversification

# In[17]:


weights = np.array([0.5, 0.5])

portfolio_return = np.dot(returns_close.mean(), weights)
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns_close.cov(), weights)))
print(f"Expected Annual Portfolio Return: {portfolio_return * 252:.2%}")
print(f"Expected Annual Portfolio Volatility: {portfolio_volatility * np.sqrt(252):.2%}")


# ## Lets add 2 more assets in our portfolio. We will add DBA for agriculture, SLV for silver

# In[18]:


symbols = ['USO', 'GLD', 'DBA', 'SLV']
data = yf.download(symbols, start='2010-01-01', end='2025-06-30')
data


# In[19]:


returns = data.pct_change().dropna()
returns


# In[20]:


weights = np.array([0.25, 0.25, 0.25, 0.25])

portfolio_return = np.dot(returns['Close'].mean(), weights)
portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(returns['Close'].cov(), weights)))
print(f"Expected Annual Portfolio Return: {portfolio_return * 252:.2%}")
print(f"Expected Annual Portfolio Volatility: {portfolio_volatility * np.sqrt(252):.2%}")

