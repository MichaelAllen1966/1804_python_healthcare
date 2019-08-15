
# coding: utf-8

# Python is a dynamic language where variable type (e.g. whether a variable is a string or an integer) is not fixed. Sometime though you might like to force a particular type, or convert between types (most common where numbers may be contained within a string.
# 
# The most common types of variables in python are integers, float (nearly all types of numbers that are not integers) and strings. 
# 
# Within python code the function <em>type</em> will show what variable type a string is. Outside of the code, within the interpreter requesting help on a variable name will give the help for its variable type. 

# In[28]:


x = 11 # This is an integer
print (type(x))


# In[29]:


help (x)


# Though x is an integer, if we divide it by 2, Python will dynamically change its type to <em>float</em>. This may be common sense to some, but may be irritating to people who are used to coding languages where variables have 'static' types (this particular behaviour is also different to Python2 where the normal behaviours is for integers to remain as integers).

# In[30]:


x = x/2
print (x)
print (type(x))


# Python is trying to be helpful, but this dynamic behaviour of variables can sometimes need a little debugging. In our example here, to keep our answer as an integer (if that is what we needed to do) we need to specify that:
# 

# In[31]:


x = 11
x = int(x/2)
print (x)
print (type(x))


# We can convert between types, such as in the follow examples. Numbers may exist as text, particularly in some file imports.

# In[32]:


x = '10'
print ('Look what happens when we try to multiply a string')
print (x *3) 
x = float (x)
print ('Now let us multiply the converted variable')
print (x*3)

