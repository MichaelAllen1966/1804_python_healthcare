
# coding: utf-8

# # Sorting and ranking with Pandas
# 
# ## Sorting
# 
# Pandas allows easy and flexible sorting.
# 
# Let's first build a dataframe:

# In[15]:


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


# And now let's sort first by magic power and then (in reverse order aggression.

# In[16]:


new_df = df.sort_values(['magic_power','aggression'], ascending=[False,True])
print (new_df)


# Usually it is fine to use the default sorting method. Sometimes though you may wish to do a series of sequential sorts where you maintain the previous order within the sorted the dataframe. In that case use a mergesort by passing <em>kind = 'mergesort'</em> as one of the arguments.

# We can use <em>sort_index</em> to sort by the index field. Let's sort our new dataframe by reverse index order:

# In[17]:


print (new_df.sort_index(ascending=False))


# ## Ranking
# 
# Pandas allows easy ranking of dataframes by a single column. Where two values are identical the result is the average of the number of ranks they would cover. Notice that a higher number is a higher rank.

# In[18]:


print (df['magic_power'].rank())


# Pandas does not offer a direct method for ranking using multiple columns. One way would be to sort the dataframe, reset the index with <em>df.reset_index()</em> and compare the index values to the original table.
