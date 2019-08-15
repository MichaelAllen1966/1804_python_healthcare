
# coding: utf-8

# # Matplotlib: simple xy line charts
# 
# Matplotlib is a powerful library for plotting data. Data may be in the form of lists, or data from NumPy and Pandas may be used directly. Charts are very highly configurable, but here we'll focus on the key basics first.
# 
# A useful resource for matplotlib is a gallery of charts with associated code:
# https://matplotlib.org/gallery.html
# 
# ## Plotting a single line
# 
# The following code plots a simple line plot and saves the file in png format. Other formats mau be used (e.g. jpg), but png offers an open source high quality format which is ideal for charts.

# In[10]:


import matplotlib.pyplot as plt

# The following line is only needed to display chart in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

X=range(100)
Y=[value ** 2 for value in X] # (A list comprehension)

plt.plot(X,Y)
plt.show()


# ## Plotting two lines
# 
# To plot two lines we simply add another plot before generating the chart with <em>plt.show()</em>

# In[11]:


import numpy as np

import matplotlib.pyplot as plt

# The following line is only needed to display chart in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

# np.linsapce creates an array of equally spaced numbers
X=np.linspace(0,2*np.pi,100) # 100 points between 0 and 2*pi
Ya=np.sin(X)
Yb=np.cos(X)

plt.plot(X,Ya)
plt.plot(X,Yb)
plt.show()


# ## Saving figures
# 
# To save a figure use <em>plt.savefig('my_figname.png')</em> before <em>plt.show()</em> tp save as png format (best for figures, but you can also use jpg or tif).
