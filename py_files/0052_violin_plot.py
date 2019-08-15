
# coding: utf-8

# # Violin plots
# 
# Violin plots are an alternative to box plots. They show the spread of data in the form of a distribution plot along the y axis. Some people love other. Others don't! See what you think.

# In[3]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

n_violins = 5
groups = np.arange(1,n_violins+1)

# Use Python list comprehension to build distributions
# Mean is i (group #), standard deviation is 0.5 * i
samples = [np.random.normal(3*i,0.5*i,250) for i in groups]

violins = plt.violinplot (samples,
                         groups,
                         points=300, # the more the smoother
                         widths=0.8,
                         showmeans=False,
                         showextrema=True,
                         showmedians=True)

# Change the bodies to grey

for v in violins['bodies']:
    v.set_facecolor('0.8')
    v.set_edgecolor('k')
    v.set_linewidth(1)
    v.set_alpha(1)
    

# Make all the violin statistics marks red:
for partname in ('cbars','cmins','cmaxes','cmedians'):
    vp = violins[partname]
    vp.set_edgecolor('r')
    vp.set_linewidth(1)
    
plt.show()

