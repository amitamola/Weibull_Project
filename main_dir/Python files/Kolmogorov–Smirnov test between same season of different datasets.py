#!/usr/bin/env python
# coding: utf-8

# ### Important libraries

# In[2]:


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


# ## Performing KS test for all the combinations

# In[3]:


import os
import itertools
import pandas as pd
from scipy.stats import kstest

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
        temp_df2 = df_loc2[df_loc2['season'] == season]

        df_ks = pd.DataFrame()
        df_ks['wdsp'] = np.sort(np.unique(np.append(temp_df1['wdsp'].values, temp_df2['wdsp'].values)))

        loc1_vals = temp_df1['wdsp'].values
        loc2_vals = temp_df2['wdsp'].values
        
        stat, p_value = kstest(loc1_vals, loc2_vals)

        df_ks['loc1'] = df_ks['wdsp'].apply(lambda x: np.mean(loc1_vals<=x))
        df_ks['loc2'] = df_ks['wdsp'].apply(lambda x: np.mean(loc2_vals<=x))

        k = np.argmax( np.abs(df_ks['loc1'] - df_ks['loc2']))
        ks_stat = np.abs(df_ks['loc2'][k] - df_ks['loc1'][k])

        y = (df_ks['loc2'][k] + df_ks['loc1'][k])/2

        plt.figure(figsize=(10,10), dpi=200)
        plt.plot('wdsp', 'loc1', data=df_ks, label=comb[0], lw=3, color='mediumseagreen')
        plt.plot('wdsp', 'loc2', data=df_ks, label=comb[1], lw=3, color='sandybrown')
        plt.errorbar(x=df_ks['wdsp'][k], y=y, yerr=ks_stat/2, color='crimson',
                    capsize=4, mew=4, label=f"Test statistic: {ks_stat:.4f}", lw=3)
        plt.legend(loc='best');
        plt.title(f"Kolmogorov-Smirnov Test : {season}", fontsize=18, pad=15);
        plt.show()

        

        # Append to final dataframe
        final_df = pd.concat([final_df, pd.DataFrame({'loc1': [comb[0]], 'loc2': [comb[1]], 'season': [season],
                                    'ks_statistic': [stat], 'p_value': [p_value]})], ignore_index=True)


# In[6]:


final_df.head()


# In[7]:


final_df.to_csv('KS_test_between_same_season_different_datasets.csv', index=False)

