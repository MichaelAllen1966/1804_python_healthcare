
# coding: utf-8

# # Confidence Interval for a single proportion 
# 
# We can use statsmodels to calculate the confidence interval of the proportion of given 'successes' from a number of trials. This may the frequency of occurrence of a gene, the intention to vote in a particular way, etc.

# In[8]:


import statsmodels.stats.proportion as smp

# e.g. 35 out of a sample 120 (29.2%) people have a particular gene type.
# What are the 95% confidence intervals on the proportion?

lower, upper = smp.proportion_confint (35, 120, alpha=0.05, method='normal')

print ('Lower confidence interval:', lower)
print ('Upper confidence interval:', upper)

