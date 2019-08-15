
# coding: utf-8

# # If, elif, else, while, and logical operators

# Like many programming languages, Python enables code to be run based on testing whether a condition is true. Python uses indented blocks to run code according to these tests.
# 
# ## If, elif, else
# 
# <em>if</em> statements run a block of code if a particular condition is true. <em>elif</em> or 'else if' statements can subsequently run to test more conditions, but these run only if none of the precious <em>if</em> or <em>elif</em> statements were true. <em>else</em> statements may be used to  run code if none of the previous <em>if</em> or <em>elif</em> were true.
# 
# <em>if</em> statements may be run alone, with <em>else</em> statements or with <em>elif</em> statements. <em>else</em> and <em>elif</em> statements require a previous <em>if</em> statement.
# 
# 
# In the following example we use the <em>input</em> command to ask a user for a password and test against known value.
# 
# Here we use just one <em>elif</em> statement, but multiple statements could be used.
# 
# Note that the tests of equality uses double = signs

# In[20]:


password = input ("Password? ") # get user to enter password
if password == "secret":
    print ("Well done")
    print ("You")
elif password=="Secret":
    print ("Close!")
else:
    print("Noodle brain")


# ## While statements
# 
# <em>while</em> statements may be used to repeat some actions until a condition is no longer true. For example: 

# In[21]:


x = 0
while x <5:
    x +=1 # This is shorthand for x = x + 1
    print (x)


# ## Logical operators
# 
# The following are commonly used logical operators:
# 
# ==  Test of identity
# 
# !=  Not equal to
# 
# \>  greater than
# 
# <  less than
# 
# \>=  equal to or greater than  
# 
# <=  less than or equal to
# 
# <em>in</em>  test whether element is in list/tuple/dictionary
# 
# <em>not in</em>  test whether element is not in list/tuple/dictionary
# 
# <em>and</em> test multiple conditions are true
# 
# <em>or</em> test one of alternative conditions are true
# 
# <em>any</em>  test whether all elements are true
# 
# <em>all</em>  test whether all elements are true
# 
# When Python test conditins they are evaluated as either True or False. These values may also be used directly in tests:

# In[22]:


x = 10
y = 20
z = 30

# using and/or
print (x>15)
print (x>15 and y>15 and z>15)
print (x>15 or y>15 or z>15)
print ()

# Using any and all
print (any([x>20, y>20, z>20]))
test_array = [x>20, y>20, z>20]
print (test_array)
print (any(test_array))
print (all(test_array))

