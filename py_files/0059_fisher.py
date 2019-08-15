
# coding: utf-8

# # Fisher's exact test
# 
# Fisher's exact test is similar to the Chi-squared test, but is suitable for small sample sizes. As a rule it should be used if at least 20% of values are less than 5 or any value is zero. Although in practice it is employed when sample sizes are small, it is valid for all sample sizes.
# 
# For example, let us look at an example where a group of 16 people may choose tennis or football. In the group of 16 there are six boys and ten girls. The tennis group has one boy and eight girls. The football group has five boys and two girls. Does the sport affect the proportion of boys and girls choosing it? 

# In[3]:


import numpy as np
import scipy.stats as stats

obs = np.array([[1,5], [8,2]])
fisher_result = stats.fisher_exact(obs)

p_val = fisher_result[1]

print ('Data:')
print (obs)

print ('\nProbability that column does not effect row values:')
print (p_val)

