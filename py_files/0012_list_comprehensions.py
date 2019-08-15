
# coding: utf-8

# # List comprehensions: one line loops
# 
# List comprehensions are a very <em>Pythonic</em> way of condensing loops  and action into a single line.

# In[1]:


my_list=[1,4,5,7,9,10,3,2,16]
my_new_list=[]
for x in my_list:
    if x>5:
        my_new_list.append(x**2)
print (my_new_list)


# This may be done by a single line list comprehension:

# In[3]:


y = [x**2 for x in my_list if x>5]
print (y)

