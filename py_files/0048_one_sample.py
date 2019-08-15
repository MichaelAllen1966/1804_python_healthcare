
# coding: utf-8

# # One sample t-test and Wilcoxon signed rank test
# 
# The following test for a difference between the centre of a sample of data and a given reference point. The one sample t-test assumes normally distributed data, whereas the Wilcoxon signed rank test can be used with any data.
# 
# ## One sample t-test

# In[8]:


import numpy as np
import scipy.stats as stats

# generate the data
normDist = stats.norm(loc=7.5, scale=3)
data = normDist.rvs(100)

# Define a value to check against
checkVal = 6.5

# T-test
# --- >>> START stats <<< ---
t, tProb = stats.ttest_1samp(data, checkVal)
# --- >>> STOP stats <<< ---

print ('P value:')
print (tProb)


# ## Wilcoxon signed rank test

# In[10]:


# Note the test value is subtracted from all data (test is then effectively against zero)

rank, pVal = stats.wilcoxon(data-checkVal)

print ('P value:')
print (pVal)

