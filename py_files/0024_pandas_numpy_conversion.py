
# coding: utf-8

# # Converting betwen NumPy and Pandas
# 
# Conversion between NumPy and Pandas is simple.
# 
# Let's start with importing NumPy and Pandas, and then make a Pandas dataframe.

# In[19]:


import numpy as np
import pandas as pd

df = pd.DataFrame()

names = ['Gandolf','Gimli','Frodo','Legolas','Bilbo']
types = ['Wizard','Dwarf','Hobbit','Elf','Hobbit']
magic = [10, 1, 4, 6, 4]
aggression = [7, 10, 2, 5, 1]
stealth = [8, 2, 5, 10, 5]


df['names'] = names
df['type'] = types
df['magic_power'] = magic
df['aggression'] = aggression
df['stealth'] = stealth

print (df)


# ## Converting from Pandas to NumPy
# 
# We will use the <em>values</em> method to convert from Pandas to NumPy. Notice that we loose our column headers when converting to a NumPy array, and the index filed (name) simply becomes the first column.

# In[20]:


my_array = df.values

print (my_array)


# ## Converting from NumPy to Pandas
# 
# We will use the dataframe method to convert from a NumPy array to a Pandas dataframe. A new index has been created, and columns have been given numerical headers. 

# In[21]:


my_new_df = pd.DataFrame(my_array)

print (my_new_df)


# If we have column names, we can supply those to the dataframe during the conversion process. We pass a list to the dataframe method:

# In[27]:


names = ['name','type','magic_power','aggression','strength']

my_new_df = pd.DataFrame(my_array, columns=names)

print(my_new_df)


# And, as we have seen previously, we can set the index to a particular column:

# In[28]:


my_new_df.set_index('name', inplace=True)

print (my_new_df)

