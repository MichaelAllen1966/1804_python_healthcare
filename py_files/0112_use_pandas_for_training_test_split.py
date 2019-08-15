#!/usr/bin/env python
# coding: utf-8

# # Splitting data set into training and test sets using Pandas DataFrames methods
# 
# Note: this may also be performed using SciKit-Learn train_test_split method, but here we will use native Pandas methods. 
# 
# ## Create a DataFrame

# In[1]:


# Create pandas data frame

import pandas as pd

name = ['Sam', 'Bill', 'Bob', 'Ian', 'Jo', 'Anne', 'Carl', 'Toni']
age = [22, 34, 18, 34, 76, 54, 21, 8]
gender = ['f', 'm', 'm', 'm', 'f', 'f', 'm', 'f']
height = [1.64, 1.85, 1.70, 1.75, 1.63, 1.79, 1.70, 1.68]
passed_physical = [0, 1, 1, 1, 0, 1, 1, 0]

people = pd.DataFrame()
people['name'] = name
people['age'] = age
people['gender'] = gender
people['height'] = height
people['passed'] = passed_physical

print(people)


# ## Split training and test sets

# Here we take a random sample (25%) of rows and remove them from the original data by dropping index values.

# In[2]:


# Create a copy of the DataFrame to work from
# Omit random state to have different random split each run

people_copy = people.copy()
train_set = people_copy.sample(frac=0.75, random_state=0)
test_set = people_copy.drop(train_set.index)

print ('Training set')
print (train_set)
print ('\nTest set')
print (test_set)
print ('\nOriginal DataFrame')
print (people)


# ## Use 'pop' to extract the label
# 
# 'Pop' will remove a column from the DataFrame, and transfer it to a new variable.

# In[3]:


train_set_labels = train_set.pop('passed')
test_set_labels = test_set.pop('passed')


# In[4]:


print ('Training set')
print (train_set)
print ('\nTraining set label (y)')
print (train_set_labels)


# In[ ]:




