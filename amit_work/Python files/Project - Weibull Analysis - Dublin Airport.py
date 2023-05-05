#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np

from matplotlib import font_manager
font_path = 'C:\\Users\\amita\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NewCM10-Regular.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


from reliability.Fitters import Fit_Weibull_3P


# # Weibull Analysis

# ## Definining Weibull function

# In[6]:


def weibull_function(alpha, beta, gamma, x):
    return  (beta/alpha) * ((x-gamma)/alpha)**(beta-1) * np.exp(-((x-gamma)/alpha)**beta)


# ## 1. For winter data

# In[7]:


season = 'winter'
data = np.loadtxt(f'{season}.txt', unpack=True, usecols=[0], skiprows=1)


# In[8]:


data


# In[5]:


season_fit = Fit_Weibull_3P(data, method='MLE', show_probability_plot = False)


# ### Plotting the weibull function using fitted parameters vs original data

# In[10]:


plt.figure(figsize=[8,5], dpi=200)

# Plotting histogram for the values
hist, bin_edges = np.histogram(data, bins=int(data.max()))
hist = hist / len(data)
speed = (bin_edges[1:] + bin_edges[:-1]) / 2

plt.bar(speed, hist, width=0.6, align='center', label='Original Data')


# Plotting weibull function
x = np.linspace(0, int(data.max()), 1000)
y = weibull_function(season_fit.alpha, season_fit.beta, season_fit.gamma, x)

plt.plot(x,y, label = f'Weibull fit', c='tab:orange', lw=2, ls='--')


# A horizontal line at y=0
plt.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.3)

# A vertical line at x=0
plt.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.3)

plt.xticks(np.arange(0,50,2))

plt.legend()
plt.title(f'Weibull Distribution for {season.capitalize()} data', fontsize=15, pad=10)
plt.show()


# In[7]:


plt.figure(figsize=[8,5], dpi=200)

# Plotting histogram for the values
hist, bin_edges = np.histogram(data, bins=int(data.max()))
hist = hist / len(data)
speed = (bin_edges[1:] + bin_edges[:-1]) / 2

plt.bar(speed, hist, width=0.6, align='center', label='Original Data')


# Plotting weibull function
x = np.linspace(0, int(data.max()), 1000)
y = weibull_function(season_fit.alpha, season_fit.beta, season_fit.gamma, x)

plt.plot(x,y, label = f'Weibull fit', c='tab:orange', lw=2, ls='--')


# A horizontal line at y=0
plt.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.3)

# A vertical line at x=0
plt.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.3)

plt.xticks(np.arange(0,50,2))

plt.legend()
plt.title(f'Weibull Distribution for {season.capitalize()} data', fontsize=15, pad=10)
plt.show()


# ### Let's make a function out of this

# In[19]:


def weibull_fitter(data, season, method='MLE'):
    season_fit = Fit_Weibull_3P(data, method=method, show_probability_plot = False)
    
    plt.figure(figsize=[8,5], dpi=200)

    # Plotting histogram for the values
    hist, bin_edges = np.histogram(data, bins=int(data.max()))
    hist = hist / len(data)
    speed = (bin_edges[1:] + bin_edges[:-1]) / 2

    plt.bar(speed, hist, width=0.6, align='center', label='Original Data')


    # Plotting weibull function
    x = np.linspace(0, int(data.max()), 1000)
    y = weibull_function(season_fit.alpha, season_fit.beta, season_fit.gamma, x)

    plt.plot(x,y, label = f'Weibull fit', c='tab:orange', lw=2, ls='--')


    # A horizontal line at y=0
    plt.axhline(y=0, color='k', linestyle='--', linewidth=2, alpha=0.3)

    # A vertical line at x=0
    plt.axvline(x=0, color='k', linestyle='--', linewidth=2, alpha=0.3)
    plt.xticks(np.arange(0,50,2))

    plt.legend()
    plt.title(f'Weibull Distribution for {season.capitalize()} data', fontsize=15, pad=10)
    plt.show()


# ## 2. For spring data

# In[10]:


season = 'spring'
data = np.loadtxt(f'{season}.txt', unpack=True, usecols=[0], skiprows=1)

weibull_fitter(data, season)


# ## 3. For summer data

# In[11]:


season = 'summer'
data = np.loadtxt(f'{season}.txt', unpack=True, usecols=[0], skiprows=1)

weibull_fitter(data, season)


# ## 4. For autumn data

# In[12]:


season = 'autumn'
data = np.loadtxt(f'{season}.txt', unpack=True, usecols=[0], skiprows=1)

weibull_fitter(data, season)


# # For whole year

# In[17]:


import pandas as pd
season = 'whole year'
data = pd.read_csv('dublin_airport_data.csv')['wdsp'].values

weibull_fitter(data, season)


# ## Daily data

# In[21]:


import pandas as pd
season = 'whole year'
data = pd.read_csv('dly532.csv')['wdsp'].values

weibull_fitter(data, season, method='MLE')

