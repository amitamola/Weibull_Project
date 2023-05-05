#!/usr/bin/env python
# coding: utf-8

# ### Important libraries

# In[2]:


import pandas as pd
import os
import itertools
from sklearn.preprocessing import StandardScaler
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import font_manager
font_path = 'C:\\Users\\amita\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NewCM10-Regular.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


# ## Performing KS test for all distributions
# 1. Removing all the values below and above 5 and 21 respectively
# 2. Standardizing the values left
# 3. Performing KS test between these new values

# In[4]:


import os
import itertools
import pandas as pd
from scipy.stats import kstest

scaler = StandardScaler()
final_df = pd.DataFrame(columns=['loc1', 'loc2', 'season', 'ks_statistic', 'p_value'])

locations = os.listdir('final_outputs')
seasons = ['winter', 'spring', 'summer', 'autumn']

# Get combination of all locations
combinations = list(itertools.combinations(locations, 2))

for comb in combinations:
    df_loc1 = pd.read_csv(f'final_outputs/{comb[0]}/{comb[0]}.csv')
    df_loc2 = pd.read_csv(f'final_outputs/{comb[1]}/{comb[1]}.csv')

    for season in seasons:
        temp_df1 = df_loc1[df_loc1['season'] == season]
        temp_df1 = temp_df1[(temp_df1['wdsp']>5) & (temp_df1['wdsp']<21)]
        temp_df1['wdsp'] = scaler.fit_transform(temp_df1['wdsp'].values.reshape(-1,1))

        temp_df2 = df_loc2[df_loc2['season'] == season]
        temp_df2 = temp_df2[(temp_df2['wdsp']>5) & (temp_df2['wdsp']<21)]
        temp_df2['wdsp'] = scaler.fit_transform(temp_df2['wdsp'].values.reshape(-1,1))

        df_ks = pd.DataFrame()
        df_ks['wdsp'] = np.sort(np.unique(np.append(temp_df1['wdsp'].values, temp_df2['wdsp'].values)))

        loc1_vals = temp_df1['wdsp'].values
        loc2_vals = temp_df2['wdsp'].values
        
        stat, p_value = kstest(loc1_vals, loc2_vals)

        df_ks['loc1'] = df_ks['wdsp'].apply(lambda x: np.mean(loc1_vals<=x))
        df_ks['loc2'] = df_ks['wdsp'].apply(lambda x: np.mean(loc2_vals<=x))

        k = np.argmax( np.abs(df_ks['loc1'] - df_ks['loc2']))
        ks_stat = np.abs(df_ks['loc2'][k] - df_ks['loc1'][k])

        # Append to final dataframe
        final_df = pd.concat([final_df, pd.DataFrame({'loc1': [comb[0]], 'loc2': [comb[1]], 'season': [season],
                                    'ks_statistic': [stat], 'p_value': [p_value]})], ignore_index=True)


# In[6]:


final_df.head()


# In[7]:


final_df.to_csv('KS_test_bw_locations_same_season_5to21.csv', index=False)

