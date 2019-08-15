
# coding: utf-8

# # Controlling decimal places in output
# 
# Python allows a lot of control over formatting of output. But here we will just look at controlling decimal places of output.
# 
# There are some different ways. Here is perhaps the most common (because it is most similar to other languages).
# 
# The number use is represented in the print function as %x.yf where x is the total number of spaces to use (if defined this will add padding to short numbers to align numbers on different lines, for example), and y is the number of decimal places. <em>f</em> informs python that we are formatting a float (a decimal number). The %x.yf is used as a placeholder in the output (multiple palceholders are possible) and the values are given at the end of the print statement as below:

# In[13]:


import math
pi = math.pi
pi_square = pi**2

print('Pi is %.3f, and Pi squared is %.3f' %(pi,pi_square))


# It is also possible to round numbers before printing (or sending to a file). If taking this approach be aware that this may limit the precision of further work using these numbers:

# In[14]:


import math
pi = math.pi
pi = round(pi,3)
print (pi)

