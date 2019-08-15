
# coding: utf-8

# # 22. Pandas basics: building a dataframe from lists, and retrieving data from the dataframe using row and column index references
# 
# Here we will repeat basic actions previously described for NumPy. There is significant overlap between NumPy and Pandas (not least because Pandas is built on top of NumPy). Generally speaking Pandas will be used more for data manipulation, and NumPy will be used more for raw calculations (but that is probably somewhat of an over-simplification!).
# 
# Pandas allows us to access data using index names or by row/column number. Using index names is perhaps more common in Pandas. You may find having the two different methods available a little confusing at first, but these dual methods are one thing that help make Pandas powerful for data manipulation.
# 
# As with NumPy, we will often be importing data from files, but here we will create a dataframe from existing lists.

# ## Creating an empty data frame and building it up from lists
# 
# We start with importing pandas (using pd as the short name we will use) and then create a dataframe.

# In[134]:


import pandas as pd
df = pd.DataFrame()


# Let's create some data in lists and add them to the dataframe:

# In[135]:


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


# We can print the dataframe. Notice that a column to the left has appeared with numbers. This is the index, which has been added automatically.

# In[136]:


print(df)


# ## Setting an index column
# 
# We can leave the index as it is, or we can make one of the columns the index. Note that to change something in an existing dataframe we use 'inplace=True'

# In[137]:


df.set_index('names', inplace=True)
print (df)


# ## Accessing data with loc and iloc
# 
# Dataframes have two basic methods of accessing data by row (or index) and by column (or header):
# 
# <em>loc</em> selects data by index name and column (header) name.
# 
# <em>iloc</em> selects data by row or column number
# 
# ## Selcting rows by index
# 
# 
# The <em>loc</em> method selects rows by index name, like in Python dictionaries:

# In[165]:


print (df.loc['Gandolf'])


# We can pass multiple index references to the loc method using a list:

# In[166]:


to_find = ['Bilbo','Gimli','Frodo']
print (df.loc[to_find])


# Row slices may also be taken. For example let us take a row slice from Gimli to Legolas. Unusually for Python this slice includes both the lower and upper index references.

# In[154]:


print (df.loc['Gimli':'Legolas'])


# As with other Python slices a colon may be used to represent the start or end. :Gimli would take a slice from the beginning to Gimli. Bilbo: would take a row slice from Bilbo to the end.

# ## Selecting records by row number
# 
# Rather than using an index, we can use row numbers, using the <em>iloc</em> method. As with most references in Python the range given starts from the lower index number and goes up to, but does not include, the upper index number.

# In[155]:


print (df.iloc[0:2])


# Discontinuous rows may be accessed with iloc by building a list:

# In[156]:


print (df.iloc[[0,1,4]])


# Or, building up a more complex list of row numbers:

# In[157]:


rows_to_find = list(range(0,2))
rows_to_find += (list(range(3,5)))


print ('List of rows to find:',rows_to_find)
print()
print (df.iloc[rows_to_find])


# ## Selecting columns ny name
# 
# Columns are selected using square brackets after the dataframe:

# In[158]:


print (df['type'])


# In[159]:


print (df[['type','stealth']])


# To take a slice of columns we need to use the <em>loc</em> method, using : to select all rows.

# In[160]:


print (df.loc[:,'magic_power':'stealth'])


# ## Selecting columns by number
# 
# Columns may also be referenced by number using the <em>column</em> method (which allows slicing):

# In[161]:


print (df[df.columns[1:4]])


# Or <em>iloc</em> may be used to select columns by number (the colon shows that we are selecting all rows):

# In[162]:


print (df.iloc[:,1:3])


# ## Selecting rows and columns simultaneously
# 
# We can combine row and column references with the <em>loc</em> method:

# In[163]:


rows_to_find = ['Bilbo','Gimli','Frodo']
print (df.loc[rows_to_find,'magic_power':'stealth'])


# Or with <em>iloc</em> (referencing row numbers):

# In[164]:


print (df.iloc[0:2,2:4])

