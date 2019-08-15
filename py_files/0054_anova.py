
# coding: utf-8

# # Analysis of variance (ANOVA)
# 
# One way analysis of variance (ANOVA) tests whether multiple groups all belong to the same population or not.
# 
# If a conclusion is reached that the groups do not all belong to the same population, further tests may be utilised to identify the differences.

# In[2]:


import numpy as np
import scipy.stats as stats

# Create four random groups of data with a mean difference of 1

mu, sigma = 10, 3 # mean and standard deviation
group1 = np.random.normal(mu, sigma, 50)

mu, sigma = 11, 3 # mean and standard deviation
group2 = np.random.normal(mu, sigma, 50)

mu, sigma = 12, 3 # mean and standard deviation
group3 = np.random.normal(mu, sigma, 50)

mu, sigma = 13, 3 # mean and standard deviation
group4 = np.random.normal(mu, sigma, 50)

# Test whether all groups belong to the same population

F_statistic, pVal = stats.f_oneway(group1, group2, group3, group4)

print ('P value:')
print (pVal)

