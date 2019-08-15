
# coding: utf-8

# # Loops and iterating

# <em>for</em> loops can be used to step through lists, tuples, and other 'iterable' objects. 
# 
# Iterating through a list:

# In[2]:


for item in [10,25,50,75,100]:
    print (item, item**2)


# A for loop may be used to generate and loop through a sequence of numbers (note that a 'range' does not include the maximum value specified):

# In[4]:


for i in range(100,150,10):
    print(i)


# A for loop may be used to loop through an index of positions in a list:

# In[8]:


my_list = ['Frodo','Bilbo','Gandalf','Gimli','Sauron']
for i in range(len(my_list)):
    print ('Index:',i,', Value',my_list[i])


# ## Breaking out of loops or continuing the loop without action 

# Though it may not be considered best coding practice, it is possible to prematurely escape a loop with the <em>break</em> command:

# In[1]:


for i in range(10): # This loop would normally go from 0 to 9
    if i == 5:
        break
    else:
        print(i)
print ('Loop complete')


# Or, rather than breaking out of a loop, it is possible to effectively skip an iteration of a loop with the <em>continue</em> command. This may be places anywhere in the loop and returns the focus to the start of the loop.

# In[2]:


for i in range (10):
    if i%2 == 0: # This is the integer remainder after dividing i by 2
        continue
    else:
        print (i)
print ('Loop complete')


# ## Using pass to replace active code in a loop

# The <em>pass</em> command is most useful as a place holder to allow a loop to be built and have contents added later.

# In[3]:


for i in range (10):
    # Some code will be added here later
    pass
print ('Loop complete')

