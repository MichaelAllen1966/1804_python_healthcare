
# coding: utf-8

# # Reshaping Pandas data with stack, unstack, pivot and melt
# 
# Sometimes data is best shaped where the data is in the form of a wide table where the description is in a column header, and sometimes it is best shaped as as having the data descriptor as a variable within a tall table. 
# 
# To begin with you may find it a little confusing what happens to the index field as we switch between different formats. But hang in there and you'll get the hang of it! 
# 
# Lets look at some examples, beginning as usual with creating a dataframe.

# In[77]:


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


# When we look at this table, the data descriptors are columns, and the data table is 'wide'.

# In[78]:


print (df)


# ## Stack and unstack
# 
# We can convert between the two formats of data with <em>stack</em> and <em>unstack</em>. To convert from a wide table to a tall and skinny, use <em>stack</em>. Notice this creates a more complex index which has two levels the first level is person id, and the second level is the data header. This is called a multi-index.

# In[79]:


df_stacked = df.stack()
print(df_stacked.head(20)) # pront forst 20 rows


# We can convert back to  wide table with <em>unstack</em>. This recreates a single index for each line of data.

# In[80]:


df_unstacked = df_stacked.unstack()
print (df_unstacked)


# Returning to our stacked data, we can convert our multi-index to two separate fields by resetting the index. By default this method names the separated index field 'level_0' and 'level_1' (multi-level indexes may have further levels as well), and the data field '0'. Let's rename them as well (comment out that row with a # to see what it would look like without renaming them). You can see the effect below:

# In[89]:


reindexed_stacked_df = df_stacked.reset_index()
reindexed_stacked_df.rename(
    columns={'level_0': 'ID', 'level_1': 'variable', 0:'value'},inplace=True)

print (reindexed_stacked_df.head(20)) # print first 20 rows


# We can return to a multi-index, if we want to, by setting the index to the two fields to be combined. Whether a multi-index is preferred or not will depend on what you wish to do with the dataframe, so it useful to know how to convert back and forth between multi-index and single-index.

# In[86]:


reindexed_stacked_df.set_index(['ID', 'variable'], inplace=True)
print (reindexed_stacked_df.head(20))


# ## Melt and pivot
# 
# <em>melt</em> and <em>pivot</em> are like <em>stack</em> and <em>unstack</em>, but offer some other options.
# 
# <em>melt</em> de-pivots data (into a tall skinny table)
# 
# <em>pivot</em> will re-pivot data into a wide table.
# 
# Let's return to our original dataframe created (which we called 'df') and  create a tall skinny table of selected fields using <em>melt</em>. We will separate out one or more of the fields, such as 'names' as an ID field, as below:

# In[87]:


unpivoted = df.melt(id_vars=['names'], value_vars=['type','magic_power'])
print (unpivoted)


# And we can use the <em>pivot</em> method to re-pivot the data, defining which field identifies the data to be grouped together, which column contains the new column headers, and which field contains the data.

# In[88]:


pivoted = unpivoted.pivot(index='names', columns='variable', values='value')
print (pivoted_2)

