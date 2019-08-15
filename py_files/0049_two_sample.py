
# coding: utf-8

# # t-tests for testing the difference between two groups of data
# 
# t-tests are ideally suited to groups of data that are normally distributed.
# 
# ## Unpaired t-test
# 
# Statistical test for testing the difference between independent groups (e.g. measure the weight of men and women).

# In[18]:


import numpy as np
import scipy.stats as stats

# Create two random groups of data with a mean difference of 1

mu, sigma = 10, 2.5 # mean and standard deviation
group1 = np.random.normal(mu, sigma, 50)

mu, sigma = 11, 2.5 # mean and standard deviation
group2 = np.random.normal(mu, sigma, 50)

# Calculate t and probability of a difference

t_statistic, p_value = stats.ttest_ind(group1, group2)

# Print results

print ('P value:')
print (p_value)


# ## Paired t-test
# 
# Statistical test for testing the difference between paired samples (e.g. measuring the weight of people at the start and at the end of an experiment).

# In[17]:


# Create a random group of data
# The add an average difference of 1 (but with variation)

mu, sigma = 10, 2.5 # mean and standard deviation
t_0 = np.random.normal(mu, sigma, 50)

mu, sigma = 1, 2.5 # mean and standard deviation
diff = np.random.normal(mu, sigma, 50)
t_1 = t_0 + diff

# Calculate t and probability of a difference

t_statistic, p_value = stats.ttest_rel(t_0, t_1)

# Print results

print ('P value:')
print (p_value)

