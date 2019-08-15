
# coding: utf-8

# # The iris data set
# 
# This is a classic 'toy' data set used for machine learning testing is the iris data set. 
# 
# The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor). Four features were measured from each sample: the length and the width of the sepals and petals, in centimetres.
# 
# It comes preloaded in scikit learn. Let's load it and have a look at it. 

# In[34]:


import numpy as np
from sklearn import datasets


iris=datasets.load_iris()


# In[35]:


# The iris dataset is an object that contains a number of elements:

print (list(iris))


# In[36]:


# feature_names shows data field titles:

print (iris.feature_names)


# In[37]:


# data is the data for each sample; columns described by feature_names:
# lets' print just the first 10 roes
print (iris.data[0:10])


# In[38]:


# target_names lists types of iris identified:

print (iris.target_names)


# In[39]:


# target lists the type of iris in each row of data:
# this maps to the target_names

print (iris.target)


# In[40]:


# DESCR gives us description of the data set:

print (iris.DESCR)


# ## Data sets in scikit learn
# 
# load_boston: boston house-prices dataset (regression).
# 
# load_iris: iris dataset (classification).
# 
# load_diabetes: diabetes dataset (regression).
# 
# load_digits: digits dataset (classification).
# 
# load_linnerud: linnerud dataset (multivariate regression).
# 
# load_wine: wine dataset (classification).
# 
# load_breast_cancer: breast cancer wisconsin dataset (classification).

# ## Other sources of test data sets
# 
# https://archive.ics.uci.edu/ml/datasets.html
# 
# https://blog.bigml.com/list-of-public-data-sources-fit-for-machine-learning/
