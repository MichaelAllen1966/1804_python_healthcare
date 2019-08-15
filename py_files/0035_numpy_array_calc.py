
# coding: utf-8

# # Array maths in NumPy
# 
# NumPy allows easy standard mathematics to be performed on arrays, a well as moire complex linear algebra such as array multiplication.
# 
# Lets begin by building a couple of arrays. We'll use the <em>np.arange</em> method to create an array of numbers in range 1 to 12, and then reshape the array into a 3 x 4 array.

# In[32]:


import numpy as np

# note that the arange method is 'half open'
# that is it includes the lower number, and goes up yo, but not including,
# the higher number

array_1 = np.arange(1,13)
array_1 = array_1.reshape (3,4)

print (array_1)


# ## Maths on a single array
# 
# We can multiple an array by a fixed number (or we can add, subtract, divide, raise to power, etc):

# In[25]:


print (array_1 *4)


# In[26]:


print (array_1 ** 0.5) # square root of array


# We can define a vector and multiple all rows by that vector:

# In[27]:


vector_1 = [1, 10, 100, 1000]

print (array_1 * vector_1)


# To multiply by a column vector we will transpose the original array, multiply by our column vector, and transpose back:

# In[28]:


vector_2 = [1, 10, 100]

result = (array_1.T * vector_2).T

print (result)


# ## Maths on two (or more) arrays
# 
# Arrays of the same shape may be multiplied, divided, added, or subtracted.
# 
# Let's create a copy of the first array:

# In[29]:


array_2 = array_1.copy()

# If we said array_2 = array_1 then array_2 would refer to array_1.
# Any changes to array_1 would also apply to array_2


# Multiplying two arrays:

# In[30]:


print (array_1 * array_2)


# ## Matrix multiplication ('dot product')
# 
# See https://www.mathsisfun.com/algebra/matrix-multiplying.html for an explanation of matrix multiplication, if you are not familiar with it.
# 
# We can perform matrix multiplication in numpy with the <em>np.dot</em> method.

# In[31]:


array_2 = np.arange(1,13)
array_2 = array_1.reshape (4,3)

print ('Array 1:')
print (array_1)
print ('\nArray 2:')
print (array_2)
print ('\nDot product of two arrays:')
print (np.dot(array_1, array_2))

