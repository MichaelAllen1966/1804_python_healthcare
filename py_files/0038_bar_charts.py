
# coding: utf-8

# # Matplotlib: Bar charts
# 
# ## A simple bar chart

# In[12]:


# The following line is only needed to display chart in Jupyter notebooks
get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt
import numpy as np

data =[5.,25.,50.,20.]
x = range(len(data)) # creates a range of 0 to 4

plt.bar(x, data)

plt.show()


# ## Multiple series

# In[13]:


import matplotlib.pyplot as plt
import numpy as np

data =[[5.,25.,50.,20],
        [4.,23.,51.,17],
        [6., 22.,52.,19]]

X=np.arange(4)

plt.bar(X+0.00,data[0],color='b',width=0.25) # colour is blue
plt.bar(X+0.25,data[1],color='g',width=0.25) # colour is green
plt.bar(X+0.50,data[2],color='r',width=0.25) # colour is red

plt.show()


# For larger data sets we could automate the creation of the bars:

# In[14]:


import matplotlib.pyplot as plt
import numpy as np

data =[[5.,25.,50.,20],
        [4.,23.,51.,17],
        [6., 22.,52.,19]]

color_list=['b','g','r']
gap=0.8/len(data)

for i, row in enumerate(data): # creates int i and list row
    X=np.arange(len(row))
    plt.bar(X+i*gap,row,width=gap,color=color_list[i])

plt.show()


# ## Stacked bar chart

# In[15]:


a=[5.,30.,45.,22.]
b=[5.,25.,50.,20.]
x=range(4)

plt.bar(x,a,color='b')
plt.bar(x,b,color='r',bottom=a) # bottom specifies the starting position for a bar

plt.show()


# ## Back to back bar chart

# In[16]:


women_pop=np.array([50,30,45,22])
men_pop=np.array([5,25,50,20])
x=np.arange(4)

plt.barh(x,women_pop,color='r')
plt.barh(x,-men_pop,color='b') # the minus sign creates an opposite bar

plt.show()

