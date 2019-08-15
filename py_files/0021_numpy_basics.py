
# coding: utf-8

# # Numpy basics: building an array from lists, basic statistics, converting to booleans, referencing the array, and taking slices
# 
# Most commonly we will be loading files into NumPy arrays, but here we build an array from lists and perform some basics stats on the array.
# 
# The examples below construct and use a 2 dimensional array (which may be though of as 'rows and columns'). Later we will look at higher dimensional arrays.

# ## Building an array from lists
# 
# We use the <em>np.array</em> function to build an array from existing lists. Here each list represents a row of a data table.

# In[108]:


import numpy as np

row0 = [23, 89,100]
row1 = [10, 51, 99]
row2 = [40, 78, 102]
row3 = [35, 81, 110]
row4 = [50, 75, 95]
row5 = [65, 51, 101]

data = np.array([row0, row1, row2, row3, row4, row5])


# We now have a data array:

# In[109]:


print (data)


# ## Performing basic statistics on the array
# 
# We can see, for example, the mean of all the data

# In[110]:


print (data.mean())


# Or we can use the <em>axis</em> argument to show the mean by column or mean by row:

# In[111]:


print (np.mean(data,axis=0)) # average all rows in each column
print (np.mean(data,axis=1)) # average all columns in each row


# Some other commonly used statsitcs (all of which are by column, or dimension 0, below):

# In[112]:


print ('Mean:\n', np.mean(data, axis=0))
print ('Median:\n', np.median(data, axis=0))
print ('Sum:\n', np.sum(data, axis=0))
print ('Maximum:\n', np.max(data,axis=0))
print ('Minimum\n:', np.min(data,axis= 0))
print ('Range:\n', np.ptp(data, axis=0))
print ('10th percentile:\n', np.percentile(data, 10, axis=0))
print ('90th percentile:\n', np.percentile(data, 90, axis=0))
print ('Population standard deviation\n' ,np.std(data, axis=0))
print ('Sample standard deviation:\n', np.std(data, axis=0, ddof=1)) 
print ('Population variance:\n', np.var(data, axis=0))
print ('Sample variance:\n', np.var(data, axis=0, ddof=1))


# The returned array may be referenced by index number (beginning at zero):

# In[113]:


results = np.mean(data, axis=0)

print (results[0])


# Basic python may also be incorporated into the statistics. For example the 10th percentiles, from 0 to 100 may be calculated in a loop (remember that the standard Python range function goes up to, but does not include, the maximum value given, so we need to put a higher maximum than 100 in order to include the 100th percentile in the loop):

# In[114]:


for percent in range(0,101,10):
    print(percent,'percentile:',np.percentile(data, percent, axis=0))


# ## Converting values to a boolean True/False
# 
# We can use test values against some standard. For example, here we test whether a value is equal to, or greater than, the mean of the column.

# In[115]:


column_mean = np.mean(data, axis=0)
greater_than_mean = data >= column_mean
print (greater_than_mean)

True/False values in Python may also be used in calculations where False has an equivalent value to zero, and True has an equivalent value to 1.
# In[116]:


print (np.mean(greater_than_mean, axis=0))


# ## Referencing arrays and taking slices
# 
# NumPy arrays are referenced similar to lists, except we have two (or more dimensions). For a two dimensional array the reference is [dimension_0, dimension 1], which is equvalent to [row, column].
# 
# REMEMBER: Like lists, indices start at index zero, and when taking a slice the slice goes up to, but does not include, the maximum value given.

# In[117]:


print ('Row zero:\n', data[0,:])
print ('Column one:\n', data[:,1])
# Note in the example below a slice goes up to ,but dot include, the maximum index
print ('Rows 0-3, Column zero:\n', data[0:4,0])
print ('Row 1, Column 2\n', data[1,2])

