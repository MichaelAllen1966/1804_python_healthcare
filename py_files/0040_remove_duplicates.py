
# coding: utf-8

# # Removing duplicate data in NumPy and Pandas
# 
# Both NumPy and Pandas offer easy ways of removing duplicate rows. Pandas offers a more powerful approach if you wish to remove rows that are partly duplicated.
# 
# ## Numpy
# 
# With numpy we use np.unique() to remove duplicate rows or columns (use the argument axis=0 for unique rows or axis=1 for unique columns).

# In[29]:


import numpy as np

array = np.array([[1,2,3,4],
                  [1,2,3,4],
                  [5,6,7,8],
                  [1,2,3,4],
                  [3,3,3,3],
                  [5,6,7,8]])

unique = np.unique(array, axis=0)
print (unique)


# We can return the index values of the kept rows with the argument return_index=True (the argument return_inverse=True would return the discarded rows):

# In[30]:


array = np.array([[1,2,3,4],
                  [1,2,3,4],
                  [5,6,7,8],
                  [1,2,3,4],
                  [3,3,3,3],
                  [5,6,7,8]])

unique, index = np.unique(array, axis=0, return_index=True)
print ('Unique rows:')
print (unique)
print ('\nIndex of kept rows:')
print (index)


# We can also count the number of times a row is repeated with the argument return_counts=True:

# In[31]:


array = np.array([[1,2,3,4],
                  [1,2,3,4],
                  [5,6,7,8],
                  [1,2,3,4],
                  [3,3,3,3],
                  [5,6,7,8]])

unique, index, count = np.unique(array, axis=0, 
                          return_index=True,
                          return_counts=True)

print ('Unique rows:')
print (unique)
print ('\nIndex of kept rows:')
print (index)
print ('\nCount of duplicate rows')
print (count)


# ## Pandas
# 
# With Pandas we use drop_duplicates.

# In[32]:


import pandas as pd
df = pd.DataFrame()

names = ['Gandolf','Gimli','Frodo', 'Gimli', 'Gimli']
types = ['Wizard','Dwarf','Hobbit', 'Dwarf', 'Dwarf']
magic = [10, 1, 4, 1, 3]
aggression = [7, 10, 2, 10, 2]
stealth = [8, 2, 5, 2, 5]


df['names'] = names
df['type'] = types
df['magic_power'] = magic
df['aggression'] = aggression
df['stealth'] = stealth


# Let's remove duplicated rows:

# In[33]:


df_copy = df.copy() # we'll work on a copy of the dataframe
df_copy.drop_duplicates(inplace=True)
print (df_copy)


# We have removed fully duplicated rows. We use inplace=True to make changes to the dataframe directly. Ulternatively we could have used new_df = df_copy.drop_duplicates(), without using inplace=True.

# We can also remove duplicates based just on a selection of columns, Here we will look for rows with duplicated names and type. Note that by default the first row is kept.

# In[34]:


df_copy = df.copy() # we'll work on a copy of the dataframe
df_copy.drop_duplicates(subset=['names','type'], inplace=True)
print (df_copy)


# We can choose to keep the last entered row with the argument keep='last':

# In[35]:


df_copy = df.copy() # we'll work on a copy of the dataframe
df_copy.drop_duplicates(subset=['names','type'], inplace=True, keep='last')
print (df_copy)


# We can also remove all duplicate rows by using the argument keep=False:

# In[36]:


df_copy = df.copy() # we'll work on a copy of the dataframe
df_copy.drop_duplicates(subset=['names','type'], inplace=True, keep=False)
print (df_copy)


# More complicated logic for choosing which record to keep would best be performed using a groupby method.
