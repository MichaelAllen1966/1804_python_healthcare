
# coding: utf-8

# # Iterating through columns and rows in NumPy and Pandas
# 
# Using <em>apply_along_axis</em> (NumPy) or <em>apply</em> is a more Pythonic way of iterating through data in NumPy and Pandas. But there may be occasions you wish to simply work your way through rows or columns in NumPy and Pandas. Here is how it is done.

# ## Numpy
# 
# NumPy is set up to iterate through rows when a loop is declared.

# In[12]:


import numpy as np

# Create an array of random numbers (3 rows, 5 columns)
array = np.random.randint(0,100,size=(3,5))

print ('Array:')
print (array)
print ('\nAverage of rows:')

# iterate through rows:
for row in array:
    print (row.mean())


# To iterate through columns we transpose the array with .T so that rows become columns (and vice versa):

# In[13]:


print('\nTransposed array:')
print (array.T)
print ('\nAverage of original columns:')

for row_t in array.T:
    print (row_t.mean())


# ## Pandas
# 
# Lets first create our data:

# In[14]:


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


# To iterate throw rows in a Pandas dataframe we use .iterrows():

# In[15]:


for index, row in df.iterrows():
    print(row[0], 'is a', row[1])


# To iterate through columns we need to do just a bit more manual work, creating a list of dataframe columns and then iterating through that list to pull out the dataframe columns:

# In[16]:


columns = list(df)
for column in columns:
    print (df[column][2]) # print the third element of the column

