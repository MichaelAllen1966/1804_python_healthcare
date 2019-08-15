
# coding: utf-8

# # A simple heatmap
# 
# Heatmaps may be generated with imshow.
# 
# We impport a colour map from the library cm.
# 
# For a list of colour maps available see: 
# https://matplotlib.org/examples/color/colormaps_reference.html

# In[5]:


# This is a simple heat map
# For list of alternative colur maps see:
# https://matplotlib.org/examples/color/colormaps_reference.html

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm # This allows different color schemes

get_ipython().run_line_magic('matplotlib', 'inline')

# Generate an array of increasing values
a=np.arange(0,400)
a = a.reshape(20,20)

# Plot the heatmap using 'inferno' from the cm colour schemes

plt.imshow(a,interpolation='nearest', cmap=cm.inferno)

# Add a scale bar

plt.colorbar()

plt.show()

