
# coding: utf-8

# # Using masks to filter data, and perform search and replace, in NumPy and Pandas
# 
# In both NumPy and Pandas we can create masks to filter data. Masks are 'Boolean' arrays - that is arrays of true and false values and provide a powerful and flexible method to selecting data.
# 
# ## NumPy
# 
# ### Creating a mask
# 
# Let's begin by creating an array of 4 rows of 10 columns of uniform random number between 0 and 100.

# In[33]:


import numpy as np

array1 = np.random.randint(0,100,size=(4,10))

print (array1)


# Now we'll create a mask to show those numbers greater than 70.

# In[34]:


mask = array1 > 70

print(mask)


# We can use that mask to extract the numbers:

# In[35]:


print (array1[mask])


# ### Using <em>any</em> and <em>all</em>
# 
# <em>any</em> and <em>all</em> allow us to check for all true or all false.
# 
# We can apply that to the whole array:

# In[36]:


print (mask.any())
print (mask.all())


# Or we can apply it column-wise (by passing <em>axis=1</em>) or row-wise (by passing <em>axis=1</em>)

# In[37]:


print ('All tests in a column are true:')
print (mask.all(axis=0))
print ('\nAny test in a row is true:')
print (mask.any(axis=1))


# We can use != to invert a mask if needed (all trues become false, and all falses become true). This can be useful, but can also become a little confusing!

# In[38]:


inverted_mask = mask!=True
print (inverted_mask)


# ### Adding or averaging trues
# 
# Boolean values (True/False) in Python also take the values 1 and 0. This can be useful for counting trues/false, for example:

# In[39]:


print ('Number of trues in array:')
print (mask.sum())


# In[40]:


print('Number of trues in array by row:')
print (mask.sum(axis=1))


# In[41]:


print('Average of trues in array by column:')
print (mask.mean(axis=0))


# ### Selecting rows or columns based on one value in that row or column

# Let's select all columns where the value of the first element is equal to, or greater than 50:

# In[42]:


mask = array1[0,:] >= 50 # colon indicates all columns, zero indicates row 0
print ('\nHere is the mask')
print (mask)
print ('\nAnd here is the mask applied to all columns')
print (array1[:,mask]) # colon represents all rows of chosen columns


# Similarly if we wanted to select all rows where the 2nd element was equal to, or greater, than 50

# In[43]:


mask = array1[:,1] >= 50 # colon indicates all roes, 1 indicates row 1 (the second row, as the first is row 0)
print ('\nHere is the mask')
print (mask)
print ('\nAnd here is the mask applied to all rows')
print (array1[mask,:]) # colon represents all rows of chosen columns


# ### Using <em>and</em> and <em>or</em>, and combining filters from two arrays

# We may create and combine multiple masks. For example we may have two masks that look for values less than 20 or greater than 80, and then combine those masks with or which is represented by | (stick).

# In[44]:


print ('Mask for values <20:')
mask1 = array1 < 20
print (mask1)

print ('\nMask for values >80:')
mask2 = array1 > 80
print (mask2)

print ('\nCombined mask:')
mask = mask1  | mask2 # | (stick) is used for 'or' with two boolean arrays
print (mask)

print ('\nSelected values using combined mask')
print (array1[mask])


# We can combine these masks in a single line

# In[45]:


mask = (array1 < 20) | (array1 > 80)
print (mask)


# We can combine masks derived from different arrays, so long as they are the same shape. For example let's produce an another array of random numbers and check for those element positions where corresponding positions of both arrays have values of greater than 50. When comparing boolean arrays we represent 'and' with &.

# In[46]:


array2 = np.random.randint(0,100,size=(4,10))

print ('Mask for values of array1 > 50:')
mask1 = array1 > 50
print (mask1)

print ('\nMask for values of array2 > 50:')
mask2 = array2 > 50
print (mask2)

print ('\nCombined mask:')
mask = mask1  & mask2 
print (mask)


# We could shorten this to:

# In[47]:


mask = (array1 > 50) & (array2 > 50)
print (mask)


# ### Setting values based on mask
# 
# We can use masks to reassign values only for elements that meet the given criteria. For example we can set the values of all cells with a value less than 50 to zero, and set all other values to 1.

# In[48]:


print ('Array at sttart:')
print (array1)
mask = array1 < 50
array1[mask] = 0
mask = mask != True # invert mask
array1[mask] = 1
print('\nNew array')
print (array1)


# We can shorten this, by making the mask implicit in the assignment command. 

# In[49]:


array2[array2<50] = 0
array2[array2>=50] = 1

print('New array2:')
print(array2)


# ### Miscellaneous examples
# 
# Select columns where the average value across the column is greater than the average across the whole array, and return both the columns and the column number.

# In[50]:


array = np.random.randint(0,100,size=(4,10))
number_of_columns = array.shape[1]
column_list = np.arange(0, number_of_columns) # create a list of column ids
array_average = array.mean()
column_average = array.mean(axis=0)
column_average_greater_than_array_average = column_average > array_average
selected_columns = column_list[column_average_greater_than_array_average]
selected_data = array[:,column_average_greater_than_array_average]

print ('Selected columns:')
print (selected_columns)
print ('\nSeelcted data:')
print (selected_data)


# ## Pandas
# 
# Filtering with masks in Pandas is very similar to numpy. It is perhaps more usual in Pandas to be creating masks testing specific columns, with resulting selection of rows. For example let's use a mask to select characters meeting conditions on majical power and aggression:

# In[51]:


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

mask = (df['magic_power'] > 3) & (df['aggression'] < 5)
print ('Mask:')
print (mask) # notice mask is a 'series'; a one dimensial DataFrame
filtered_data = df[mask] # when passing a Boolean series to a dataframe we select the appropriate rows
print (filtered_data)


# Though creating masks based on particular columns will be most common in Pandas. We can also filter on the entire dataframe. Look what happens when we filter on values > 3:

# In[52]:


mask = df > 3
print('Mask:')
print (mask)
print ('\nMasked data:')
df2 = df[mask]
print (df2)


# The structure of the dataframe is maintained, and all text is maintained. Those values not >3 have been removed (NaN represents 'Not a Number').

# ### Conditional replacing of values in Pandas
# 
# Replacing values in Pandas, based on the current value, is not as simple as in NumPy. For example, to replace all values in a given column, given a conditional test, we have to (1) take one column at a time, (2) extract the column values into an array, (3) make our replacement, and (4) replace the column values with our adjusted array.
# 
# For example to replace all values less than 4 with zero (in our numeric columns):

# In[53]:


columns = ['magic_power','aggression','stealth'] # to get a list of all columns you can use list(df)

for column in columns: # loop through our column list
    values = df[column].values # extract the column values into an array
    mask = values < 4 # create Boolean mask 
    values [mask] = 0 # apply Boolean mask
    df[column] = values # replace the dataframe column with the array
    
print (df)

