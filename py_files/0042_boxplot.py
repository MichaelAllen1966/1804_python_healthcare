
# coding: utf-8

# # Boxplots in matplotlib 
# 
# Matplotlib allows easy creation of boxplots. These traditionally show median (middle line across box), uper and lower quartiles (box), range excluding outliers (whiskers) and outliers (points). The default setting for outliers is points more than 1.5xIQR above or below the quartiles.
# 
# *IQR = inter-quartile range.

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')


import matplotlib.pyplot as plt
import numpy as np

x=np.random.randn(500) # samples from a normal distribution

plt.boxplot(x)

plt.show()


# ## Plotting groups
# 
# Boxplot can take data from multiple columns in a NumPy array.

# In[2]:


x=np.random.randn(1000,5) # samples from a normal distribution

plt.boxplot(x)

plt.show()


# Or data may come from separate sources:

# In[5]:


x1=list(np.random.randn(100)*5)
x2=list((np.random.randn(50)*2)+5)
x3=list(np.random.randn(250)*3)
x4=list((np.random.randn(70)*10)-10)

x=[x1,x2,x3,x4]

plt.boxplot(x)

plt.show()

