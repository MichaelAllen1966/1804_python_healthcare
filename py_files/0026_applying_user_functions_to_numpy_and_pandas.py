
# coding: utf-8

# # Applying user-defined functions to NumPy and Pandas
# 
# Both NumPy and Pandas allow user to functions to applied to all rows and columns (and other axes in NumPy, if multidimensional arrays are used)
# 
# ## Numpy
# 
# In NumPy we will use the <em>apply_along_axis</em> method to apply a user-defined function to each row and column.
# 
# Let's first set up an array and define a function. We will use a simple user-defined function for illustrative purposes - one that returns the position of the highest value in the slice passed to the function. In NumPy we use <em>argmax</em> for finding the position of the highest values.

# In[26]:


import numpy as np
import pandas as pd

my_array = np.array([[10,2,13],
                     [21,22,23],
                     [31,32,33],
                     [10,57,20],
                     [20,20,20],
                     [101,91,10]])


def my_function(x):
    position = np.argmax(x)
    return position


# Using <em>axis=0</em> we can apply that function to all columns:

# In[27]:


print (np.apply_along_axis(my_function, axis=0, arr=my_array))


# Using <em>axis=1</em> we can apply that function to all rows:

# In[28]:


print (np.apply_along_axis(my_function, axis=1, arr=my_array))


# ## Pandas
# 
# Pandas has a similar method, the <em>apply</em> method for applying a user function by either row or column. The Pandas method for determining the position of the highest value is <em>idxmax</em>. 
# 
# We will convert our NumPy array to a Pandas dataframe, define our function, and then apply it to all columns. Notice that becase we are working in Pandas the returned value is a Pandas series (equivalent to a DataFrame, but with one one axis) with an index value. 

# In[29]:


import pandas as pd

df = pd.DataFrame(my_array)

def my_function(x):
    z= x.idxmax()
    return z

print(df.apply(my_function, axis=0))


# And applying it to all rows:

# In[30]:


print(df.apply(my_function, axis=1))

