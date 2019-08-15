
# coding: utf-8

# # Linear regression with scipy.stats

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

get_ipython().run_line_magic('matplotlib', 'inline')

# Set up x any arrays

x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([2.3,4.5,5.0,8,11.1,10.9,13.9,15.4,18.2,19.5])
y=y+10

# scipy linear regression

gradient, intercept, r_value, p_value, std_err = stats.linregress(x,y)

# Calculated fitted y

y_fit=intercept + (x*gradient)

# Plot data

plt.plot(x, y, 'o', label='original data')
plt.plot(x, y_fit, 'r', label='fitted line')

# Add text box and legend

text='Intercept: %.1f\nslope: %.2f\nR-square: %.3f' %(intercept,gradient,r_value**2)
plt.text(6,15,text)
plt.legend()

# Display plot

plt.show()

