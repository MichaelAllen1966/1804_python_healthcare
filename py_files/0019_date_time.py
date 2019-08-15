
# coding: utf-8

# # Accessing date and time, and timing code
# 
# It may sometimes be useful to access the current date and/or time. As an example, when writing code it may be useful to access the time at particular stages to monitor how long different parts of code are taking.
# 
# To access date and time we will use the <em>datetime</em> module (which is held in a package that is also, a little confusingly called <em>datetime</em>!):

# In[76]:


from datetime import datetime


# To access the current date and time we can use the <em>now</em> method:

# In[77]:


print (datetime.now())


# You might not have expected to get the time to the nearest microsecond! But that may be useful at times when timing short bits of code (which may have to run many times).
# 
# But we can access the date and time in different ways:

# In[78]:


current_datetime = datetime.now()
print ('Date:', current_datetime.date()) # note the () after date
print ('Year:', current_datetime.year)
print ('Month:', current_datetime.month)
print ('Day:', current_datetime.day)
print ('Time:', current_datetime.time()) # note the () after time
print ('Hour:', current_datetime.hour)
print ('Minute:', current_datetime.minute)
print ('Second:', current_datetime.second)


# Having microseconds in the time may look a little clumsy. So let's format the date and using the <em>strftime</em> method.

# In[79]:


now = datetime.now()
print (now.strftime('%Y-%m-%d'))
print (now.strftime('%H:%M:%S'))


# ## Timing code
# 
# We can use datetime to record the time before and after some code to caclulate the time taken, but it is a little simpler to use the <em>time</em> module which keeps all time in seconds (from January 1st 1970) and gives even more accuracy to the time:

# In[81]:


import time
time_start = time.time()
for i in range(100000):
    x = i ** 2
time_end = time.time()
time_taken = time_end - time_start
print ('Time taken:', time_taken)

