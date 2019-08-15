
# coding: utf-8

# # Adding error bars to charts
# 
# ## Adding error bars to line plots

# In[4]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# example data
x = np.arange(0.1, 10, 1)
y = x ** 2

# calculate example errors (could also be from list or NumPy array)
lower_y_error = y * 0.2
upper_y_error = y * 0.3
y_error = [lower_y_error, upper_y_error]
lower_x_error = x * 0.05
upper_x_error = x * 0.05
x_error = [lower_x_error, upper_x_error]

# To use only x or y errors simple omit the relevant argument

plt.errorbar(x, y, yerr = y_error, xerr = x_error, fmt='-o')

plt.show()


# ## Adding error bars to bar charts

# In[5]:


import matplotlib.pyplot as plt
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')

x=[1,2,3,4]
y=[5.,25.,50.,20.]
yerr_upper=[2,10,5,3]
yerr_lower=np.zeros(len(yerr_upper))

fig, ax = plt.subplots()
ax.bar(x,y,
        width=0.8,
        color='0.5',
        edgecolor='k',
        yerr=[yerr_lower,yerr_upper],
        linewidth = 2,
        capsize=10)

ax.set_xticks(x)
ax.set_xticklabels(('G1', 'G2', 'G3', 'G4'))

plt.savefig('plot_21.png')

plt.show()

