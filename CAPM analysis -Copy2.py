#!/usr/bin/env python
# coding: utf-8

# # APPLICATION OF CAPITAL ASSET PRICING MODEL
# ### EMPIRICAL EVIDENCES FROM INDIAN STOCK EXCHANGE
# 
# AUTHOR: SANJEEV SINGH

# In[1]:


import pandas as pd
import seaborn as sns
import plotly.express as px
from copy import copy
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import plotly.figure_factory as ff
import plotly.graph_objects as go


# In[2]:


#Read the stock data file
stocks_df = pd.read_csv('CAPM_Stocks.csv')
stocks_df.head()


# In[3]:


stocks_df.tail(8)


# In[4]:


# Calculate correlation coefficients
correlation = stocks_df.corr()


print("Correlation Coefficients:")
print(correlation)


# In[5]:


# Function to normalize the prices based on the initial price
def normalize(df):
  x = df.copy()
  for i in x.columns[1:]:
    x[i] = x[i]/x[i][0]
  return x


# In[6]:


# Function to plot interactive plot
def interactive_plot(df, title):
  fig = px.line(title = title)
  for i in df.columns[1:]:
    fig.add_scatter(x = df['Date'], y = df[i], name = i)
  fig.show()


# In[7]:


# Plot interactive chart
interactive_plot(stocks_df, 'Prices')


# In[8]:


# Plot normalized interactive chart
interactive_plot(normalize(stocks_df), 'Normalized Prices')


# # CALCULATE MONTHLY RETURNS

# In[9]:


# Function to calculate the monthly returns 
def monthly_return(df):

  df_monthly_return = df.copy()
  
  # Loop through each stock
  for i in df.columns[1:]:
    
    # Loop through each row belonging to the stock
    for j in range(1, len(df)):
      
      # Calculate the percentage of change from the previous day
      df_monthly_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1]) * 100
    
    # set the value of first row to zero, as previous value is not available
    df_monthly_return[i][0] = 0
  return df_monthly_return


# In[10]:


# Get the monthly returns 
stocks_monthly_return = monthly_return(stocks_df)
stocks_monthly_return.head()


# In[11]:


stocks_monthly_return.mean()


# In[12]:


# Nifty 50 average monthly return is 0.89%
# Adani Enterprises average monthly return is 4.98%
# TCS average monthly return is 1.16%


# # CALCULATE BETA FOR A SINGLE STOCK

# In[13]:


# Select any stock, let's say TCS 
stocks_monthly_return['TCS']


# In[14]:


# Select the Nifty50 (Market)
stocks_monthly_return['Nifty 50'].head()


# In[15]:


# plot a scatter plot between the selected stock and the Nifty 50 (Market)
stocks_monthly_return.plot(kind = 'scatter', x = 'Nifty 50', y = 'TCS',figsize=(8, 8),color='green')
plt.show()


# In[16]:


# Fit a polynomial between the selected stock and the Nifty 50 (Poly with order = 1 is a straight line)

# beta represents the slope of the line regression line (market return vs. stock return). 
# Beta is a measure of the volatility or systematic risk of a security or portfolio compared to the entire market (Nifty 50) 
# Beta is used in the CAPM and describes the relationship between systematic risk and expected return for assets 

# Beta = 1.0, this indicates that its price activity is strongly correlated with the market. 
# Beta < 1, indicates that the security is theoretically less volatile than the market (Ex: Utility stocks). If the stock is included, this will make the portfolio less risky compared to the same portfolio without the stock.
# Beta > 1, indicates that the security's price is more volatile than the market. 
# Tech stocks generally have higher betas than Nifty 50 but they also have excess returns
#Alpha (α): Alpha represents the intercept of the regression line and indicates the excess return (positive or negative) that a stock generates compared to its expected return based on its beta and the market return.
#It represents the stock's performance independent of the market. A positive alpha suggests that the stock has outperformed the market, while a negative alpha indicates underperformance.

beta, alpha = np.polyfit(stocks_monthly_return['Nifty 50'], stocks_monthly_return['TCS'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('TCS', round(beta,3), round(alpha,3)))


# In[17]:


#TCS_returns = 0.665 * Nifty50_returns + 0.574
#Here:

#0.665 is the beta coefficient, which indicates that for every 1% change in the 'Nifty 50' returns, 'TCS' returns change by approximately 0.665%.
#0.574 is the y-intercept, representing the expected 'TCS' returns when 'Nifty 50' returns are zero.


# In[18]:


beta, alpha = np.polyfit(stocks_monthly_return['Nifty 50'], stocks_monthly_return['Sbi Bank'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('Sbi Bank', round(beta,3), round(alpha,3)))


# In[19]:


#1.476 (the first element):
#This is the beta coefficient (also referred to as the beta value) for SBI Bank stock. It measures the sensitivity of the stock's returns to changes in the market index returns, specifically the 'Nifty 50' index returns. A beta greater than 1 suggests that the stock is more volatile than the market, and a 1.476 beta indicates that for every 1% change in the 'Nifty 50' index returns, SBI Bank stock returns change by approximately 1.476%. In simple terms, SBI Bank's stock tends to move more than the market.

#alpha = -0.187 (the second element):
#This is the y-intercept of the linear regression line. In this context, it represents the expected returns of SBI Bank stock when the 'Nifty 50' index returns are zero. A negative y-intercept indicates that the stock's expected returns are negative when the market returns are zero.


# In[20]:


# Now let's plot the scatter plot and the straight line on one plot
stocks_monthly_return.plot(kind = 'scatter', x = 'Nifty 50', y = 'TCS',figsize=(8, 8))

# Straight line equation with alpha and beta parameters 
# Straight line equation is y = beta * rm + alpha
plt.plot(stocks_monthly_return['Nifty 50'], beta * stocks_monthly_return['Nifty 50'] + alpha, '--', color = 'r')
plt.show()


# In[21]:


beta, alpha = np.polyfit(stocks_monthly_return['Nifty 50'], stocks_monthly_return['Adani Enterprises'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('Adani Enterprises', round(beta,3), round(alpha,3))) 


# In[22]:


# Now let's plot the scatter plot and the straight line on one plot
stocks_monthly_return.plot(kind = 'scatter', x = 'Nifty 50', y = 'Adani Enterprises',figsize=(8, 8))

# Straight line equation with alpha and beta parameters 
# Straight line equation is y = beta * rm + alpha
plt.plot(stocks_monthly_return['Nifty 50'], beta * stocks_monthly_return['Nifty 50'] + alpha, '-', color = 'g')
plt.show()


# ##  APPLY THE CAPM FORMULA TO AN INDIVIDUAL STOCK

# In[23]:


#Adani Enterprises
beta 


# In[24]:


# Let's calculate the average monthly rate of return for Nifty 50
stocks_monthly_return['Nifty 50'].mean()


# In[25]:


# Let's calculate the annualized rate of return for Nifty 50 
stocks_monthly_returns = [0.8859]

mean_return = np.mean(stocks_monthly_returns)  # Calculate the mean return
annualized_return = mean_return * 12  # Multiply by 12 to annualize the return

# Round the result to 3 decimal places
annualized_return = round(annualized_return, 3)
annualized_return


# In[26]:


# Also you can use the yield of a 10-years Indian Government bond as a risk free rate 7.076
rf = 7.076

# Calculate return for any security (Adani Enterprises) using CAPM  
ER_Adani_Enterprises = round(rf + ( beta * (annualized_return-rf) ) ,3)
ER_Adani_Enterprises


# In[27]:


# You have to calculate Beta for ITC first
beta, alpha = np.polyfit(stocks_monthly_return['Nifty 50'], stocks_monthly_return['ITC'], 1)
print('Beta for {} stock is = {} and alpha is = {}'.format('ITC', round(beta,3), round(alpha,3)))


# In[28]:


# Calculate return for ITC using CAPM  
ER_ITC = round(rf + ( beta * (annualized_return - rf) ),3) 
print(ER_ITC)


# In[29]:


# Let's do the same plots but in an interactive way

for i in stocks_monthly_return.columns:
  
  if i != 'Date' and i != 'Nifty 50':
    
    # Use plotly express to plot the scatter plot for every stock vs. the Nifty 50
    fig = px.scatter(stocks_monthly_return, x = 'Nifty 50', y = i, title = i)

    # Fit a straight line to the data and obtain beta and alpha
    b, a = np.polyfit(stocks_monthly_return['Nifty 50'], stocks_monthly_return[i], 1)
    
    # Plot the straight line 
    fig.add_scatter(x = stocks_monthly_return['Nifty 50'], y = b*stocks_monthly_return['Nifty 50'] + a)
    fig.show()


# ## CALCULATE BETA FOR ALL STOCKS

# In[60]:


# Let's view Beta for every stock 
beta


# In[61]:


# Let's view alpha for each of the stocks
# Alpha describes the strategy's ability to beat the market (Nifty 50)
# Alpha indicates the “excess return” or “abnormal rate of return,” 
# A positive 0.133 alpha for ITC means that the portfolio’s return exceeded the benchmark Nifty 50 index by 13.3%.
alpha


# In[32]:


# Let's create a placeholder for all betas and alphas (empty dictionaries)
beta = {}
alpha = {}

# Loop on every stock monthly return
for i in stocks_monthly_return.columns:

  # Ignoring the date and Nifty 50 Columns 
  if i != 'Date' and i != 'Nifty 50':
    # plot a scatter plot between each individual stock and the Nifty 50 (Market)
    stocks_monthly_return.plot(kind = 'scatter', x = 'Nifty 50', y = i,figsize=(8, 8))
    
    # Fit a polynomial between each stock and the Nifty 50 (Poly with order = 1 is a straight line)
    b, a = np.polyfit(stocks_monthly_return['Nifty 50'], stocks_monthly_return[i], 1)
    
    plt.plot(stocks_monthly_return['Nifty 50'], b * stocks_monthly_return['Nifty 50'] + a, '-', color = 'r')
    
    beta[i] = b
    
    alpha[i] = a
    plt.show()


# In[33]:


# Obtain a list of all stock names
keys = list(beta.keys())
keys


# In[34]:


# Let's calculate the annualized rate of return for Nifty 50 
stocks_monthly_returns = [0.8859]  # Replace [...] with the actual list of monthly returns

mean_return = np.mean(stocks_monthly_returns)  # Calculate the mean return
rm = mean_return * 12  # Multiply by 12 to annualize the return

# Round the result to 3 decimal places
rm = round(rm, 3)
rm


# In[35]:


# Define the expected return dictionary
ER = {}

rf = 7.076
rm = round(stocks_monthly_return['Nifty 50'].mean() * 12,3) # this is the expected return of the market 
rm


# In[36]:


for i in keys:
# Calculate return for every security using CAPM  
  ER[i] = rf + ( beta[i] * (rm-rf) )


# In[37]:


for i in keys:
  print('Expected Return Based on CAPM for {} is {}%'.format(i, round(ER[i],3)))


# In[38]:


# Assume equal weights in the portfolio
portfolio_weights = 1/25 * np.ones(25) 
portfolio_weights


# In[39]:


# Calculate the portfolio return 
ER_portfolio_all = round(sum(list(ER.values()) * portfolio_weights),3)
print('Expected Return Based on CAPM for the portfolio  is {}%\n'.format(ER_portfolio_all))
print('Suggested to invest, Returns better than Nifty 50')


# In[40]:


# Calculate the portfolio return of auto sector
ER_portfolio_auto_sector = round(0.333 * ER['M&M'] +  0.333 * ER['Maruti']+0.333 * ER['Tata Motors'],3)
print('Expected Return Based on CAPM for the portfolio (Auto Sector) is {}%\n'
      .format(ER_portfolio_auto_sector))
print('Suggested to invest, Returns better than Nifty 50')


# In[41]:


# Calculate the portfolio return of banking sector
ER_portfolio_banking_sector = round(0.333 * ER[' HDFC Bank'] +  0.333 * ER['Kotak Bank']+0.333 * ER['Sbi Bank'],3)
print('Expected Return Based on CAPM for the portfolio (banking Sector) is {}%\n'
      .format(ER_portfolio_banking_sector))
print('Suggested to invest, Returns better than Nifty 50')


# In[42]:


# Calculate the portfolio return of consumer durable sector
ER_portfolio_consumer_durable_sector = round(0.5 * ER['Trent'] + 0.5 * ER['Asian Paint'],3)
print('Expected Return Based on CAPM for the portfolio (consumer durable) is {}%\n'
      .format(ER_portfolio_consumer_durable_sector))
print('Suggested not to invest, Returns worse than Nifty 50')


# In[43]:


# Calculate the portfolio return of IT sector
ER_portfolio_IT_sector = round(0.5 * ER['TCS'] + 0.5 * ER['Wipro'],3)
print('Expected Return Based on CAPM for the portfolio (IT) is {}%\n'
      .format(ER_portfolio_IT_sector))
print('Suggested not to invest, Returns worse than Nifty 50')


# In[44]:


# Calculate the portfolio return of Real Estate sector
ER_portfolio_Real_Estate_sector = round(0.5 * ER['DLF'] + 0.5 * ER['Godrej Prop'],3)
print('Expected Return Based on CAPM for the portfolio (Real Estate) is {}%\n'
      .format(ER_portfolio_Real_Estate_sector))
print('Suggested to invest, Returns better than Nifty 50')


# In[45]:


# Calculate the portfolio return of Energy sector
ER_portfolio_Energy_sector = round(0.5 * ER['Reliance'] + 0.5 * ER['Adani Enterprises'],3)
print('Expected Return Based on CAPM for the portfolio (Energy ) is {}%\n'
      .format(ER_portfolio_Energy_sector))
print('Suggested to invest, Returns better than Nifty 50')


# In[46]:


# Calculate the portfolio return of Financial sector
ER_portfolio_Financial_sector = round(0.5 * ER['Bajaj Finance'] + 0.5 * ER['LIC Housing Finance'],3)
print('Expected Return Based on CAPM for the portfolio (Financial ) is {}%\n'
      .format(ER_portfolio_Financial_sector))
print('Suggested to invest, Returns better than Nifty 50')


# In[47]:


# Calculate the portfolio return of Derived Material sector
ER_portfolio_Derived_Material_sector = round(0.333 * ER['Asian Paint'] +  0.333 * ER['Jindal Stainless']+0.333 * ER['Tata Steel'],3)
print('Expected Return Based on CAPM for the portfolio (Derived Material) is {}%\n'
      .format(ER_portfolio_Derived_Material_sector))
print('Suggested to invest, Returns better than Nifty 50')


# In[48]:


# Calculate the portfolio return of health sector
ER_portfolio_health_sector = round(0.333 * ER['Apollo Hospital'] +  0.333 * ER['Dr Reddy lab']+0.333 * ER['Sun Pharma'],3)
print('Expected Return Based on CAPM for the portfolio (health Sector) is {}%\n'
      .format(ER_portfolio_health_sector))
print('Suggested not to invest, Returns worse than Nifty 50')


# In[49]:


# Calculate the portfolio return of FMCG sector
ER_portfolio_FMCG_sector = round(0.333 * ER['ITC'] +  0.333 * ER['Nestle']+0.333 * ER['HUL'],3)
print('Expected Return Based on CAPM for the portfolio (FMCG Sector) is {}%\n'
      .format(ER_portfolio_FMCG_sector))
print('Suggested not to invest, Returns worse than Nifty 50')


# In[50]:


# Calculate the portfolio return of High performing(10) stocks
ER_portfolio_hp = round(0.1 * ER['M&M'] + 0.1 * ER['Tata Motors'] + 0.1 * ER['Sbi Bank'] + 0.1 * ER['Jindal Stainless'] + 0.1 * ER['Tata Steel'] + 0.1 * ER['Adani Enterprises'] + 0.1 * ER['Bajaj Finance'] + 0.1 * ER['LIC Housing Finance'] + 0.1 * ER['DLF'] + 0.1 * ER['Godrej Prop'], 3)
print('Expected Return Based on CAPM for the portfolio (High performing) is {}%\n'.format(ER_portfolio_hp))
print('Suggested to invest, Returns better than Nifty 50')


# In[51]:


# Calculate the portfolio return of weak performing(9) stocks
ER_portfolio_lp = round(0.111 * ER['Asian Paint'] + 0.111 * ER['HUL'] + 0.111 * ER['ITC'] + 0.111 * ER['Nestle'] + 0.111 * ER['Apollo Hospital'] + 0.111 * ER['Dr Reddy lab'] + 0.111 * ER['Sun Pharma'] + 0.111 * ER['TCS'] + 0.111 * ER['Wipro'], 3)
print('Expected Return Based on CAPM for the portfolio (weak performing) is {}%\n'.format(ER_portfolio_lp))
print('Suggested not to invest, Returns worse than Nifty 50')


# In[52]:


# Calculate the portfolio return of Average performing(6) stocks
ER_portfolio_ap = round(0.166 * ER['Maruti'] + 0.166 * ER[' HDFC Bank'] + 0.166 * ER['Kotak Bank'] + 0.166 * ER['Havells'] + 0.166 * ER['Trent'] + 0.166 * ER['Reliance'], 3)
print('Expected Return Based on CAPM for the portfolio (Average performing) is {}%\n'.format(ER_portfolio_ap))
print('No suggestion because Nifty 50 return and Average performing stocks portfolio return are nearly same')


# In[53]:


# Calculate the portfolio return of extremes stocks (r>12.50)
ER_portfolio_ed = round(0.166 * ER['Tata Motors'] + 0.166 * ER['Jindal Stainless'] + 0.166 * ER['Adani Enterprises'] + 0.166 * ER['Bajaj Finance'] + 0.166 * ER['DLF'] + 0.166 * ER['Godrej Prop'], 3)
print('Expected Return Based on CAPM for the portfolio (Extremes) is {}%\n'
      .format(ER_portfolio_ed))


# In[54]:


# Calculate the portfolio of stocks that return above than market return
ER_portfolio_am = round(.066 * ER['M&M'] +  .066 * ER['Maruti']+ 0.066 * ER['Tata Motors'] + .066 * ER[' HDFC Bank']+ 
                      .066 * ER['Sbi Bank']+ .066* ER['Havells'] + .066* ER['Trent']+  .066 * ER['Jindal Stainless']+ 0.066 * ER['Tata Steel'] + .066 * ER['Adani Enterprises']+ 
                     0.066 * ER['Reliance'] +  .066 * ER['Bajaj Finance']+ .066* ER['LIC Housing Finance'] + .066* ER['DLF']+ .066 * ER['Godrej Prop'],3)
print('Expected Return Based on CAPM for the portfolio (Above than  Market Return) is {}%\n'
      .format(ER_portfolio_am))


# In[55]:


sum=0
for i in keys:
    sum=sum+ER[i] 
for i in keys:
  print('Contribution on CAPM for {} is {}'.format(i, round(ER[i]/sum,2)))


# In[56]:


ER_portfolio_b = round(.04 * ER['M&M'] +  .04 * ER['Maruti']+ 0.05 * ER['Tata Motors'] + .04 * ER[' HDFC Bank']+ 
                     0.04 * ER['Kotak Bank'] +  .05 * ER['Sbi Bank']+ .04* ER['Havells'] + .04* ER['Trent']+ .04 * ER['Asian Paint'] +  .05 * ER['Jindal Stainless']+ 0.04 * ER['Tata Steel'] + .05 * ER['Adani Enterprises']+ 
                     0.04 * ER['Reliance'] +  .05 * ER['Bajaj Finance']+ .04* ER['LIC Housing Finance'] + .03* ER['HUL']+ .04 * ER['ITC'] +  .03 * ER['Nestle']+ 0.04 * ER['Apollo Hospital'] + .03 * ER['Dr Reddy lab']+ 
                     0.03 * ER['Sun Pharma'] +  .03 * ER['TCS']+ .03* ER['Wipro'] + .05* ER['DLF']+ .05 * ER['Godrej Prop'],3)
print('Expected Return Based on CAPM for the portfolio (Balanced Weightage) is {}%\n'
      .format(ER_portfolio_b))


# In[57]:


data = {
    'Combinations': ['M&M', 'Maruti', 'Tata Motors', 'HDFC Bank', 'Kotak Bank', 'SBI Bank', 'Havells', 'Trent', 'Asian Paint', 'Jindal Stainless',
                     'Tata Steel', 'Adani Enterprises', 'Reliance', 'Bajaj Finance', 'LIC Housing Finance', 'HUL', 'ITC', 'Nestle', 'Apollo Hospital',
                     'Dr Reddy lab', 'Sun Pharma', 'TCS', 'Wipro', 'DLF', 'Godrej Prop', 'Equal Portfolio Weights', 'Auto Sector', 'Banking Sector',
                     'Consumer Durable Sector', 'IT Sector', 'Real Estate', 'Energy', 'Financial', 'Derived Material', 'Health Sector', 'FMCG',
                     'High Performing', 'Weak Performing', 'Average Performing', 'Extremes', 'Above than Market Return', 'Balanced Weightage'],
    'Expected Return Based on CAPM': [11.424, 10.929, 13.398, 10.81, 10.47, 12.324, 10.66, 10.705, 9.49, 12.528, 11.881, 13.51, 10.864, 13.34, 11.44,
                                      8.177, 9.691, 8.773, 9.837, 7.728, 9.054, 9.44, 8.902, 12.651, 12.507, 10.822, 11.905, 11.192,
                                      10.098, 9.171, 12.579, 12.187, 12.39, 11.288, 8.864, 8.871, 12.5, 9.001, 10.698, 12.937, 11.812, 11.203]
}

Returns = pd.DataFrame(data)
Returns.sort_values(by='Expected Return Based on CAPM', ascending=False, inplace=True)
print(Returns)


# In[58]:


print("Top 10 Suggested Portfolio") 
Returns[:10]


# In[59]:


print("Bottom 5 Suggested Portfolio") 
Returns.tail()


# In[ ]:




