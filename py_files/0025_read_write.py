
# coding: utf-8

# # Reading and writing CSV files using NumPy and Pandas
# 
# Here we will load a CSV called iris.csv. This is stored in the same directory as the Python code.
# 
# As a general rule, using the Pandas import method is a little more 'forgiving', so if you have trouble reading directly into a NumPy array, try loading in a Pandas dataframe and then converting to a NumPy array. We will load two files into NumPy and Pandas, iris.csv and iris_numbers.csv. Both of these files may be found at: https://github.com/MichaelAllen1966/wordpress_blog/tree/master/jupyter_notebooks

# ## Reading a csv file into a NumPy array
# 
# NumPy's <em>loadtxt</em> method reads delimited text. We specify the separator as a comma. The data we are loading also has a text header, so we use <em>skiprows=1</em> to skip the header row, which would cause problems for NumPy.

# In[9]:


import numpy as np

my_array = np.loadtxt('iris_numbers.csv',delimiter=",", skiprows=1)

print (my_array[0:5,:]) # first 5 rows


# ## Saving a NumPy array as a csv file
# 
# We use the <em>savetxt</em> method to save to a csv. 

# In[12]:


np.savetxt("saved_numpy_data.csv", my_array, delimiter=",")


# ## Reading a csv file into a Pandas dataframe
# 
# The <em>read_csv</em> will read a CSV into Pandas. This import assumes that there is a header row. If there is no header row, then the argument <em>header = None</em> should be used as part of the command. Notice that a new index column is created.

# In[23]:


import pandas as pd

df = pd.read_csv('iris.csv')

print (df.head(5)) #  First 5 rows


# ## Saving a Pandas dataframe to a CSV file
# 
# The <em>to_csv</em> will save a dataframe to a CSV. By default column names are saved as a header, and the index column is saved. If you wish not to save either of those use <em>header=True</em> and/or <em>index=True</em> in the command. For example, in the command below we save the dataframe with headers, but not with the index column.

# In[26]:


df.to_csv('my_pandas_dataframe.csv', index=False)

