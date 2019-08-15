
# coding: utf-8

# # Plotting histograms with matplotlib and NumPy
# 
# Matplotlib has an easy method for plotting data. NumPy has an easy method for obtaining histogram data.
# 
# ## Plotting histograms with Matplotlib
# 
# Plotting a histogram with a defined number of bins:

# In[18]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

x=np.random.randn(1000) # samples from a normal distribution

plt.hist(x,bins=20)
plt.title ('Defined number of bins')
plt.xlabel ('x')
plt.ylabel ('Count')

plt.show()


# Plotting a histogram with a defined range of bins:

# In[19]:


x=np.random.randn(1000) # samples from a normal distribution

# Use np.arange to create bins from -4 to +4 in steps of 0.5
# A custom list could also be used

plt.hist(x,bins=np.arange(-4,4.5,0.5))

plt.title ('Defined bin range and width')
plt.xlabel ('x')
plt.ylabel ('Count')

plt.show()


# ## Obtaining histogram data with NumPy
# 
# If histogram data is needed in addition to, or instead of, a plot, NumPy may be used. Here a defined number of bins is used:

# In[20]:


import numpy as np
count, bins = np.histogram(x, bins=20)
print ('Bins:')
print (bins)
print ('\nCount:')
print (count)


# And here a defined bin range is used.

# In[21]:


import numpy as np
count, bins = np.histogram(x, bins=np.arange(-5,6,1))
print ('Bins:')
print (bins)
print ('\nCount:')
print (count)

