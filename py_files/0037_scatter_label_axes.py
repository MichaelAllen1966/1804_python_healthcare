
# coding: utf-8

# # Scatter plot, and labelling axes

# In[1]:


# The following line is only needed to display chart in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np

# Create a 1024 by 2 array of random points
data=np.random.rand(1024,2)

x = data[:,0]
y = data[:,1]
plt.scatter(x, y)

plt.xlabel ('Series 1')
plt.ylabel ('Series 2')

plt.show()

