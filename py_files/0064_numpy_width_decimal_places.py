
# coding: utf-8

# # Setting width and number of decimal places in NumPy print output
# 
# The function below may be used to set both the number of decimal places and the fixed width of NumPy print out. If no width is given it defaults to zero (no extra padding added). Setting width will ensure alignment of output.

# In[34]:


import numpy as np

# We can set NumPy to print to a fixed number of decimal places:
# e.g. the 8 decimal places that are standard, with no set width

np.set_printoptions(formatter={'float': lambda x: "{0:0.8f}".format(x)})

# Or to make things more flexible/clearer in the code we can set up a function.
# This function will define both fixed with and decimal placed/
# If no width is given it will default to zero, with no extra spacing added

def set_numpy_decimal_places(places, width=0):
    set_np = '{0:' + str(width) + '.' + str(places) + 'f}'
    np.set_printoptions(formatter={'float': lambda x: set_np.format(x)})


# Let's generate some random number data:

# In[35]:


x = np.random.rand(3,3)


# The array as originally printed:

# In[36]:


print (x)


# Setting the number of decimal places:

# In[37]:


set_numpy_decimal_places(3)
print (x)


# Setting the number of decimal places and a fixed with:

# In[38]:


set_numpy_decimal_places(3, 6)
print (x)

