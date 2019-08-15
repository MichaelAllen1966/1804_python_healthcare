
# coding: utf-8

# # Creating a grid of subplots
# 
# There are various ways of creating subplots in Matplotlib.
# 
# Here we will use add_subplot to bring four plots together.
# 
# It is also worth looking at <em>subplot2grid</em> if you want plots of different sizes bough together.

# In[1]:


import numpy as np
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

# Define the size of the overall figure
fig = plt.figure(figsize=(8,8)) # 8 inch * 8 inch

# Create subplot 1

ax1 = fig.add_subplot(221) # Grid of 2x2, this is suplot 1

x=range(100)
y=[value ** 2 for value in x]

ax1.plot(x,y)

ax1.set_xlabel('x')
ax1.set_ylabel('x squared')
ax1.set_title('Subplot 1: A line chart')

# Create subplot 2

ax2 = fig.add_subplot(222) # Grid of 2x2, this is suplot 2

data=np.random.rand(1024,2)

ax2.scatter(data[:,0],data[:,1])

ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Subplot 2: A Scatter plot')

# Create subplot 3

ax3 = fig.add_subplot(223) # Grid of 2x2, this is suplot 3

data=[5.,25.,50.,20.]

ax3.bar(range(len(data)),data)

ax3.set_xlabel('Group')
ax3.set_ylabel('Number of patients')
ax3.set_title('Subplot 3: A bar chart')

# Create subplot 4

ax4 = fig.add_subplot(224) # Grid of 2x2, this is suplot 4

labels = 'Dan','Sean','Andy','Mike','Kerry'
cake_consumption = [10, 15, 12, 30, 100]

ax4.pie(cake_consumption, labels=labels)

ax4.set_title ("Subplot 4: A pie chart")

# Add an overall title

plt.suptitle('Four subplots', size = 20)

# Adjust the spacing between plots

plt.tight_layout(pad=4)

# Show plot

plt.show()

