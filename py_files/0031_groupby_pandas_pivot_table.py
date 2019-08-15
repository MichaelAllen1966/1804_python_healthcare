
# coding: utf-8

# # Summarising data by groups in Pandas using pivot_tables and groupby
# 
# Pandas offers two methods of summarising data - groupby and pivot_table*. The data produced can be the same but the format of the output may differ. 
# 
# *pivot_table summarises data. There is a similar command, pivot, which we will use in the next section which is for reshaping data.
# 
# As usual let's start by creating a dataframe.

# In[81]:


import pandas as pd
df = pd.DataFrame()

names = ['Gandolf',
         'Gimli',
         'Frodo',
         'Legolas',
         'Bilbo',
         'Sam',
         'Pippin',
         'Boromir',
         'Aragorn',
         'Galadriel',
         'Meriadoc']
types = ['Wizard',
         'Dwarf',
         'Hobbit',
         'Elf',
         'Hobbit',
         'Hobbit',
         'Hobbit',
         'Man',
         'Man',
         'Elf',
         'Hobbit']
magic = [10, 1, 4, 6, 4, 2, 0, 0, 2, 9, 0]
aggression = [7, 10, 2, 5, 1, 6, 3, 8, 7, 2, 4]
stealth = [8, 2, 5, 10, 5, 4 ,5, 3, 9, 10, 6]


df['names'] = names
df['type'] = types
df['magic_power'] = magic
df['aggression'] = aggression
df['stealth'] = stealth


# ## Pivot tables
# 
# To return the median values by type:

# In[82]:


import numpy as np # we will use a numpy method to summarise data

pivot = df.pivot_table(index=['type'],
                        values=['magic_power','aggression', 'stealth'],
                        aggfunc=[np.mean],
                        margins=True) # margins summarises all

# note: addgunc can be any valid function that can act on data provided

print (pivot)


# Or we may group by more than one index. In this case we'll return the average and summed values by type and magical power:

# In[83]:


pivot = df.pivot_table(index=['type','magic_power'],
                        values=['aggression', 'stealth'],
                        aggfunc=[np.mean,np.sum],
                        margins=True) # margins summarises all

print (pivot)


# ## Groupby
# 
# Grouby is a very powerful method in Pandas which we shall return to in the next section. Here we will use groupby simply to summarise data.

# In[84]:


print(df.groupby('type').median())


# Instead of built in methods we can also apply user-defined functions. To illustrate we'll define a simple function to return the lower quartile.

# In[85]:


def my_func(x):
    return (x.quantile(0.25))

print(df.groupby('type').apply(my_func))

# Note we need not apply a lambda function
# We may apply any user-defined function


# As with pivot-table we can have more than one index column.

# In[86]:


print(df.groupby(['type','magic_power']).median())


# Or we can return just selected data columns.

# In[87]:


print(df.groupby('type').median()[['magic_power','stealth']])


# To return multiple types of results we use the <em>agg</em> argument.

# In[88]:


print(df.groupby('type').agg([min, max]))


# ### Pandas built-in groupby functions 
# 
# Remember that <em>apply</em> can be used to apply any user-defined function
# 
# .all # Boolean True if all true
# 
# .any # Boolean True if any true
# 
# .count count of non null values
# 
# .size size of group including null values
# 
# .max
# 
# .min
# 
# .mean
# 
# .median
# 
# .sem
# 
# .std
# 
# .var
# 
# .sum
# 
# .prod
# 
# .quantile
# 
# .agg(functions) # for multiple outputs
# 
# .apply(func)
# 
# .last # last value
# 
# .nth # nth row of group
