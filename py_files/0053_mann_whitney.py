
# coding: utf-8

# # Mann-Whitney U test
# 
# The Mann-Whitney U test allows comparison of two groups of data where the data is not normally distributed.

# In[4]:


import numpy as np
import scipy.stats as stats

# Create two groups of data

group1 = [1, 5 ,7 ,3 ,5 ,8 ,34 ,1 ,3 ,5 ,200, 3]
group2 = [10, 18, 11, 12, 15, 19, 9, 17, 1, 22, 9, 8]

# Calculate u and probability of a difference

u_statistic, pVal = stats.mannwhitneyu(group1, group2)

# Print results

print ('P value:')
print (pVal)

