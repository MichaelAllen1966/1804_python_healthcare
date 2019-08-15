
# coding: utf-8

# # Matplotlib: Pie charts
# 
# As with all matplotlib charts, pie charts are highly configurable. Here we just show the simplest of pie charts.

# In[4]:


# The following line is only needed to display chart in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

labels = 'Dan','Sean','Andy','Mike','Kerry'

cake_consumption = [10, 15, 12, 30, 100]

plt.pie(cake_consumption, labels=labels)
plt.title ("PenCHORD's Cake Consumption")

plt.show()

