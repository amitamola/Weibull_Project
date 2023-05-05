#!/usr/bin/env python
# coding: utf-8

# ### Important libraries

# In[13]:


import pandas as pd
import os
import itertools
import numpy as np

import matplotlib.pyplot as plt

from matplotlib import font_manager
font_path = 'C:\\Users\\amita\\AppData\\Local\\Microsoft\\Windows\\Fonts\\NewCM10-Regular.otf'  # Your font path goes here
font_manager.fontManager.addfont(font_path)
prop = font_manager.FontProperties(fname=font_path)

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = prop.get_name()


# ## Doing for all the combinations

# In[14]:


import os
import itertools
from scipy.stats import kstest

final_df = pd.DataFrame(columns=['location', 'season1', 'season2', 'ks_statistic', 'p_value'])

locations = os.listdir('final_outputs')
seasons = ['winter', 'spring', 'summer', 'autumn']

# Get combination of all locations
combinations = list(itertools.combinations(seasons, 2))

for loc in locations:
    df_loc = pd.read_csv(f'final_outputs/{loc}/{loc}.csv')

    for comb in combinations:
        temp_df1 = df_loc[df_loc['season'] == comb[0]]
        temp_df2 = df_loc[df_loc['season'] == comb[1]]

        df_ks = pd.DataFrame()
        df_ks['wdsp'] = np.sort(np.unique(np.append(temp_df1['wdsp'].values, temp_df2['wdsp'].values)))

        loc1_vals = temp_df1['wdsp'].values
        loc2_vals = temp_df2['wdsp'].values
        
        stat, p_value = kstest(loc1_vals, loc2_vals)

        df_ks['season1'] = df_ks['wdsp'].apply(lambda x: np.mean(loc1_vals<=x))
        df_ks['season2'] = df_ks['wdsp'].apply(lambda x: np.mean(loc2_vals<=x))

        k = np.argmax( np.abs(df_ks['season1'] - df_ks['season2']))
        ks_stat = np.abs(df_ks['season2'][k] - df_ks['season1'][k])

        y = (df_ks['season2'][k] + df_ks['season1'][k])/2

        plt.figure(figsize=(10,10), dpi=200)
        plt.plot('wdsp', 'season1', data=df_ks, label=comb[0], lw=3, color='mediumseagreen')
        plt.plot('wdsp', 'season2', data=df_ks, label=comb[1], lw=3, color='sandybrown')
        plt.errorbar(x=df_ks['wdsp'][k], y=y, yerr=ks_stat/2, color='crimson',
                    capsize=4, mew=4, label=f"Test statistic: {ks_stat:.4f}", lw=3)
        plt.legend(loc='best');
        plt.title(f"Kolmogorov-Smirnov Test : {loc}", fontsize=18, pad=15);
        plt.show()

        

        # Append to final dataframe
        final_df = pd.concat([final_df, pd.DataFrame({'location': [loc], 'season1': [comb[0]], 'season2': [comb[1]],
                                    'ks_statistic': [stat], 'p_value': [p_value]})], ignore_index=True)
        


# In[15]:


final_df.head()


# In[16]:


final_df.to_csv('KS_test_within_dataset_bw_seasons.csv', index=False)


# # Making RidgePlot for different Location's seasons

# In[10]:


from ridge_plot import joyplot
import seaborn as sns
locations = os.listdir('final_outputs')

for loc in locations:
    df = pd.read_csv(f'final_outputs/{loc}/{loc}.csv')
    
    joyplot(df, by='season', column='wdsp', figsize=(10,7), ylabelsize=16, x_range=range(-1,40), 
        colormap=sns.color_palette("crest", as_cmap=True));
    plt.xlabel('Wind Speed', fontsize=16);
    plt.title(f"Distribution for different seasons - {loc}", fontsize=24)
    plt.show();
    print('\n\n\n')

