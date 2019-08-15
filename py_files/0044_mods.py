
# coding: utf-8

# # Common modifications to charts
# 
# Here we show some common modifications to charts. These include:
# 
# * Changing scatter plot point style
# * Changing line plot line and marker style
# * Adding a legend
# * Adding some text
# * Changing axis scales
# * Changing axis ticks
# * Adding a grid
# * Adding axis tiles 
# * Adding chart title
# 

# In[48]:


import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

get_ipython().run_line_magic('matplotlib', 'inline')

data=np.random.rand(100,2)

# To give us maximum control over axes we set up the figure in this way:

ax1 = plt.figure().add_subplot(111)

# Add scatter plot
# Adjust scatter plot points by shape (marker), size (s),
# color, and edgecolour. Add a label for legend.

ax1.scatter(data[:,0],data[:,1],
            marker='s',
            s = 20,
            color = 'w',
            edgecolors = 'k',
            label  = 'scatter points')
           
x = np.linspace(0,1,100)
y = np.linspace(0,1,100)

# Add line plot
# Adjust line for colour, style, and width, and marker
# shape, frequency, and coloring. Colours may be given by letter
# or by a number between '0.0' (black) and '1.0' (white)
# Add a label for legend.

ax1.plot(x,y,
        color = 'r',
        linestyle = '--',
        linewidth = 5, 
        marker = 'o', 
        markevery = 10,
        markersize=9,
        markeredgewidth=1.5,
        markerfacecolor='0.75',
        markeredgecolor='k',
        label  = 'line')

# Adjust axes limits

ax1.set_xlim(-0.2,1.2)
ax1.set_ylim(-0.2,1.2)

# Adjust axes tickmarks

ax1.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax1.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
ax1.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax1.yaxis.set_minor_locator(ticker.MultipleLocator(0.1))

# Add a grid

ax1.grid(True, which='both') # which may be major, minor or both

# Add axis titles

ax1.set_xlabel('x', size = 15)
ax1.set_ylabel('y', size = 15)
# Add a title

ax1.set_title ('Modifying a chart', size = 20)

# Add some text at a given position

plt.text(0.7,1.1,'I am some text')

# Add the legend

ax1.legend() # see help (plt.legend) for options

plt.show()

