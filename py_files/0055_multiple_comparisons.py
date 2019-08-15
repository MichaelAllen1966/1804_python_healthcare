
# coding: utf-8

# # Multiple comparisons between groups
# 
# If an ANOVA test has identified that not all groups belong to the same population, then methods may be used to identify which groups are significantly different to each other. 
# 
# Below are two commonly used methods: Tukey's and Holm-Bonferroni.
# 
# These two methods assume that data is approximately normally distributed.
# 
# ## Setting up the data, and running an ANOVA

# In[24]:


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

# Show the results for Anova

F_statistic, pVal = stats.f_oneway(group1, group2, group3, group4)

print ('P value:')
print (pVal)


# For the multicomparison tests we will put the data into a dataframe. And then reshape it to a stacked dataframe

# In[25]:


# Put into dataframe

df = pd.DataFrame()
df['treatment1'] = group1
df['treatment2'] = group2
df['treatment3'] = group3
df['treatment4'] = group4

# Stack the data (and rename columns):

stacked_data = df.stack().reset_index()
stacked_data = stacked_data.rename(columns={'level_0': 'id',
                                            'level_1': 'treatment',
                                            0:'result'})
# Show the first 8 rows:

print (stacked_data.head(8))


# ## Tukey's multi-comparison method
# 
# See https://en.wikipedia.org/wiki/Tukey's_range_test
# 
# This method tests at P<0.05 (correcting for the fact that multiple comparisons are being made which would normally increase the probability of a significant difference being identified). A results of 'reject = True' means that a significant difference has been observed.

# In[26]:


from statsmodels.stats.multicomp import (pairwise_tukeyhsd,
                                         MultiComparison)

# Set up the data for comparison (creates a specialised object)
MultiComp = MultiComparison(stacked_data['result'],
                            stacked_data['treatment'])

# Show all pair-wise comparisons:

# Print the comparisons

print(MultiComp.tukeyhsd().summary())


# ## Holm-Bonferroni Method
# 
# See: https://en.wikipedia.org/wiki/Holm%E2%80%93Bonferroni_method
# 
# The Holm-Bonferroni method is an alterantive method.

# In[27]:


comp = MultiComp.allpairtest(stats.ttest_rel, method='Holm')
print (comp[0])

