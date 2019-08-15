
# coding: utf-8

# Here we are going to use the math module as a an untroduction to using modules. The math module contains a range of useful mathemetical functions that are not built into Python directly. So let's go ahead and start by importing the module. Module imports are usually performed at th start of a programme.

# In[13]:


import math


# When this type of import is used Python loads up a link to the module so that module functions and variables may be used. Here for example we access the value of pi through the module.

# In[14]:


print (math.pi)


# Another way of accessing module contents is to directly load up a function ror a variable into Python. When we do this we no longer need to use the module name after the import. This method is not generally recommended as it can lead to conflicts of names, and is not so clear where that function or variable comes from. But here is how it is done.

# In[15]:


from math import pi
print (pi)


# Multiple methods and variables may be loaded at the same time in this way.

# In[16]:


from math import pi, tau, log10
print (tau)
print (log10(100))


# But usually it is better practice to keep using the the library name in the code.

# In[17]:


print (math.log10(100))


# To access help on any Python function use the help command in a Python interpreter.

# In[18]:


help (math.log10)


# To find out all the methods in a module, and how to use those methods we can simply type help (module_name) into the Python interpreter. The module must first have been imported, as we did for math above.

# In[19]:


help (math)


# So now, for example, we know that to take a square root of a number we can use the math module, and use the sqrt() function, or use the pow() function which can do any power or root.

# In[20]:


print (math.sqrt(4))
print (math.pow(4,0.5))


# In Python you might read about packages as well as modules. The two names are sometimes used interchangeably, but strictly a package is a collection of modules.
