#!/usr/bin/env python
# coding: utf-8

# # 118: Python basics - saving python objects to disk with pickle
# 
# Sometimes we may wish to save Python objects to disc (for example if we have performed a lot of processing to get to a certain point). We can use Python's pickle method to save and reload any Python object. Here we will save and reload a NumPy array, and then save and reload a collection of different objects.

# ## Saving a single python object
# 
# Here we will use pickle to save a single object, a NumPy array.

# In[1]:


import pickle 
import numpy as np

# Create array of random numbers:
my_array= np.random.rand(2,4)
print (my_array)


# In[2]:


# Save using pickle
filename = 'pickled_array.p'
with open(filename, 'wb') as filehandler:
    pickle.dump(my_array, filehandler)


# Reload and print pickled array:

# In[3]:


filename = 'pickled_array.p'
with open(filename, 'rb') as filehandler: 
    reloaded_array = pickle.load(filehandler)

print ('Reloaded array:')
print (reloaded_array)


# ## Using a tuple to save multiple objects
# 
# We can use pickle to save a collection of objects grouped together as a list, a dictionary, or a tuple. Here we will save a collection of objects as a tuple.

# In[4]:


# Create an array, a list, and a dictionary
my_array = np.random.rand(2,4)
my_list =['A', 'B', 'C']
my_dictionary = {'name': 'Bob', 'Age': 42}


# In[5]:


# Save all items in a tuple
items_to_save = (my_array, my_list, my_dictionary)
filename = 'pickled_tuple_of_objects.p'
with open(filename, 'wb') as filehandler:
    pickle.dump(items_to_save, filehandler)


# Reload pickled tuple, unpack the objecys, and print them.

# In[6]:


filename = 'pickled_tuple_of_objects.p'
with open(filename, 'rb') as filehandler:
    reloaded_tuple = pickle.load(filehandler)


reloaded_array = reloaded_tuple[0]
reloaded_list = reloaded_tuple[1]
reloaded_dict = reloaded_tuple[2]

print ('Reloaded array:')
print (reloaded_array)
print ('\nReloaded list:')
print (reloaded_list)
print ('\n Reloaded dictionary')
print (reloaded_dict)

