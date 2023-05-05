#!/usr/bin/env python
# coding: utf-8

# ### Import the libraries

# In[49]:


import pandas as pd
import os
from reliability.Fitters import Fit_Weibull_3P

import matplotlib.pyplot as plt
import numpy as np


# ### Function to run the analysis on multiple dataset in one go

# In[53]:


col_names = ('season', 'month', 'year', 'alpha', 'beta', 'gamma', 'mean', 'median', 'variance', 'prob_btw_5_to_21')

# Define a function to map the month to season
def get_season(month):
    if month in [3, 4, 5]:
        return 'spring'
    elif month in [6, 7, 8]:
        return 'summer'
    elif month in [9, 10, 11]:
        return 'autumn'
    else:
        return 'winter'

def get_results(filename, est_method='LS'):
    ### Getting the dataset
    df = pd.read_csv(f'Data/{filename}.csv', parse_dates=['date'])
    df['wdsp'].fillna('0', inplace=True)
    df['wdsp'].replace(' ', '0', inplace=True)
    df['wdsp'] = df['wdsp'].astype('int64')
    
    ### Reducing the dataframe between the two dates i.e., 10 years

    ### 1. Keep only values that are after 2011-01-01
    df = df[(df['date'] > '2011-01-01') & (df['date']<'2023-01-01')]
    df = df.reset_index(drop=True)
    
    # Filling empty values by interpolation
    df['wdsp'] = df['wdsp'].interpolate()
    
    ### Creating year, month and season column    
    # Create a new column 'year' to extract the year from the date
    df['year'] = df['date'].dt.year

    # Create a new column 'month' to extract the month from the date
    df['month'] = df['date'].dt.month

    # Create a new column 'season' by mapping the month to season
    df['season'] = df['date'].dt.month.apply(get_season)
    
    ### Saving the new file
    if not os.path.exists(f'final_outputs/{filename}/'):
        os.makedirs(f'final_outputs/{filename}/', exist_ok=True)
        print(f'{filename} folder created')
    df.to_csv(f'final_outputs/{filename}/{filename}.csv', index=False)
    
    ### Fitting Weibull distribution and finding parameters seasons in total
    ## 1. Just seasons
    print("Starting analysis for all seasons!")

    entries = []

    ye = 'all'
    mo = 'all'

    for se in ['winter', 'spring', 'summer', 'autumn']:
        datum = df[(df['season']==se)]['wdsp'].values
        season_fit = Fit_Weibull_3P(datum, method=est_method, show_probability_plot = False, print_results=False)    
        prob_bw = season_fit.distribution.CDF(xvals=21)-season_fit.distribution.CDF(xvals=5)

        entries.append((se, mo, ye, season_fit.alpha, season_fit.beta, season_fit.gamma, 
                        season_fit.distribution.mean, season_fit.distribution.median, 
                        season_fit.distribution.variance, prob_bw))

    out_df = pd.DataFrame(entries, columns=col_names)
    out_df.to_csv(f'final_outputs/{filename}/{filename}_all.csv', index=False)
    print("Done for all seasons!\n")

    ## 2. Year wise seasons
    print("Starting analysis for yearly seasons!")
    
    entries = []
    mo = 'all'

    for ye in range(2013, 2023):
        for se in ['winter', 'spring', 'summer', 'autumn']:
            datum = df[(df['season']==se) & (df['year']==ye)]['wdsp'].values
            season_fit = Fit_Weibull_3P(datum, method=est_method, show_probability_plot = False, print_results=False)    
            prob_bw = season_fit.distribution.CDF(xvals=21)-season_fit.distribution.CDF(xvals=5)

            entries.append((se, mo, ye, season_fit.alpha, season_fit.beta, season_fit.gamma, 
                            season_fit.distribution.mean, season_fit.distribution.median, 
                            season_fit.distribution.variance, prob_bw))

    out_df = pd.DataFrame(entries, columns=col_names)
    out_df.to_csv(f'final_outputs/{filename}/{filename}_yearly_season.csv', index=False)
    print("Done for yearly seasons!\n")

    ## 3. Year wise months
    print("Starting analysis for yearly month wise!")

    entries = []
    season_dict = {'winter':(12,1,2), 'spring':(3,4,5), 'summer':(6,7,8), 'autumn':(9,10,11)}

    for ye in range(2013, 2023):
        for se in ['winter', 'spring', 'summer', 'autumn']:
            for mo in season_dict[se]:
                datum = df[(df['season']==se) & (df['year']==ye) & (df['month']==mo)]['wdsp'].values
                season_fit = Fit_Weibull_3P(datum, method=est_method, show_probability_plot = False, print_results=False)    
                prob_bw = season_fit.distribution.CDF(xvals=21)-season_fit.distribution.CDF(xvals=5)

                entries.append((se, mo, ye, season_fit.alpha, season_fit.beta, season_fit.gamma, 
                                season_fit.distribution.mean, season_fit.distribution.median, 
                                season_fit.distribution.variance, prob_bw))

    out_df = pd.DataFrame(entries, columns=col_names)
    out_df.to_csv(f'final_outputs/{filename}/{filename}_yearly_month.csv', index=False)
    print("Done for yearly month wise!\n")


# ### Running the code for all the datasets

# In[54]:


for f_name in os.listdir('Data'):
    print(f"Getting results for {f_name} dataset")
    get_results(f_name.split('.')[0])
    print("Done\n")

