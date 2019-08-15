
# coding: utf-8

# # Else after while
# 
# The <em>else</em> clause is only executed when your while condition becomes false. If you break out of the loop it won't be executed.

# The following loop will run the <em>else</em> code:

# In[4]:


x = 0
while x < 5:
    x += 1
    print (x)
else:
    print ('Else statement has run')
print ('End')


# The following loop with a <em>break</em> will not run the <em>else</em> code:

# In[5]:


x = 0
while x < 5:
    x += 1
    print (x)
    if x >3:
        break
else:
    print ('Else statement has run')
print ('End')

