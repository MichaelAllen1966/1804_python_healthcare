
# coding: utf-8

# # Adding data to NumPy and Pandas
# 
# ## Numpy
# 
# ### Adding more rows
# 
# To add more rows to an existing numpy array use the <em>vstack</em> method which can add multiple or single rows. New data may be in the form of a numpy array or a list. All combined data must have the same number of columns.

# In[9]:


import numpy as np

# Starting with a NumPy array
array1 = np.array([[1,2,3,4,5],
         [6,7,8,9,10],
         [11,12,13,14,15]])

# An additional 2d list
array2 = [[16,17,18,19,20],
         [21,22,23,24,25]]

# An additional single row Numpy array
array3 = np.array([26,27,28,29,30])

# We will combine all data into existing array, array1
# But a new name could be given
array1 = np.vstack([array1, array2, array3])

print (array1)


# ### Adding more columns of data
# 
# To add more columns to an existing numpy array use the <em>hstack</em> method which can add multiple or single rows. All combined data must have the same number of rows.

# In[10]:


import numpy as np

# Start with a numpy array
array1 = np.array([[1,2],
         [6,7],
         [11,12]])

# an additional multi-row numpy array
array2 = np.array([[3,4],
         [8,9],
         [13,14]])
# an additional single column list
# Note: the vertical appearance is for easy of reading only
# The square bracketed values within a wider set of square brackets will set this as a column
array3 = [[5],
         [10],
         [15]]

array1 = np.hstack([array1, array2, array3])

print (array1)


# ## Pandas
# 
# 

# ### Adding more rows of data
# 
# Here we will use the <em>concat</em> method to add more rows. Note that we have to define column names for the rows we will be adding.
# 
# Notice what happens to the index column on the left, and the order of the columns

# In[11]:


import pandas as pd

df1 =pd.DataFrame()

# Building an initial dataframe from individual lists:

names = ['Gandolf','Gimli']
types = ['Wizard','Dwarf']
magic = [10, 1]
aggression = [7, 10,]
stealth = [8, 2]

df1['names'] = names
df1['type'] = types
df1['magic_power'] = magic
df1['aggression'] = aggression
df1['stealth'] = stealth

# We can also define a dataframe with lists of all data for each row,
# but we need to remember to pass column names, as a list, to the dataframe

col_names = ['names','type','magic_power','aggression','stealth']

df2 = pd.DataFrame(
    [['Frodo','Hobbit',4,2,5],
     ['Legolas','Elf',6,5,10]],
        columns = col_names)

df1 = pd.concat([df1,df2])
print (df1)


# Each dataframe had indexes starting with zero, and those numbers are kept when combining the dataframes. This may be approproate if the index column are unique identifiers, but with a numbered index we may prefer to let the index of the appended dataframe be ignored, and the index allowed to continue its original order. We do this by passing <em>ignore_index = True</em> to the concat method.

# In[12]:


import pandas as pd

df1 =pd.DataFrame()

# Building an initial dataframe from individual lists:

names = ['Gandolf','Gimli']
types = ['Wizard','Dwarf']
magic = [10, 1]
aggression = [7, 10,]
stealth = [8, 2]

df1['names'] = names
df1['type'] = types
df1['magic_power'] = magic
df1['aggression'] = aggression
df1['stealth'] = stealth

# We can also define a dataframe with lists of all data for each row,
# but we need to remember to pass column names, as a list, to the dataframe

col_names = ['names','type','magic_power','aggression','stealth']

df2 = pd.DataFrame(
    [['Frodo','Hobbit',4,2,5],
     ['Legolas','Elf',6,5,10]],
        columns = col_names)

df1 = pd.concat([df1,df2],ignore_index = True)
print (df1)


# In the above examples the concat method has reordered columns (there is another method, append, which does not reorder columns, but append is less efficient for combining larger dataframes). To re-order columns we can pass the column order to the new dataframe. Thois could be done by appending [col names] to the end of the concat statement, or mayy be performed as a separate step:

# In[13]:


col_names = ['names','type','magic_power','aggression','stealth']
df1 = df1[col_names]
print(df1)


# ### Adding more columns of data
# 
# Individual columns of data may be added to a dataframe simply by defining a new column and passing a list of values to it.

# In[14]:


df1 = pd.DataFrame()
names = ['Gandolf','Gimli','Frodo','Legolas','Bilbo']
types = ['Wizard','Dwarf','Hobbit','Elf','Hobbit']

df1['names'] = names
df1['type'] = types

print (df1)

# Add another column
magic = [10, 1, 4, 6, 4]
df1['magic'] = magic

print ('\n Added column:\n',df1)


# We can use <em>concat</em> also to add multiple columns (in the form of another dataframe), in which case the data will be combined based on the index column. We pass the argument <em>axis=1</em> to the <em>concat</em> statement to instruct the method to combine by column (it defaults to axis=0, or row concatenation).

# In[15]:


df1 = pd.DataFrame()
names = ['Gandolf','Gimli','Frodo','Legolas','Bilbo']
types = ['Wizard','Dwarf','Hobbit','Elf','Hobbit']

df1['names'] = names
df1['type'] = types

print (df1)

df2 = pd.DataFrame()

magic = [10, 1, 4, 6, 4]
aggression = [7, 10, 2, 5, 1]
stealth = [8, 2, 5, 10, 5]

df2['magic_power'] = magic
df2['aggression'] = aggression
df2['stealth'] = stealth


# In[16]:


df1 = pd.concat([df1,df2], axis=1)
print(df1)


# There is more information here: https://pandas.pydata.org/pandas-docs/stable/merging.html
