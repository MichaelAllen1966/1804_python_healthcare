
# coding: utf-8

# ## Continuing a statement across multiple lines
# 
# Some Python editors will 'soft wrap' long lines - that is that they will split a line of code according to the window size. Not all editors do this however and a beeter practice is to split long lines into multiple lines of code. Depending on custom the line may be split when it exceeds 80, 100 or 120 characters (and many editors will allow a guide to be shown at any desired length.
# 
# There are two ways of continuing code on two lines. The first, usually preferred, method is to encase the code in curved brackets. The second method is put a back-slash outside of quotes.

# In[6]:


# This method of continuation across multiple lines is usually preferred
x = (100 + 200 +300 + 400 +
    500 + 600 + 700 + 800)
print (x)

# This is an alterantive method
x = 100 + 200 + 300 + 400 + 500 + 600 + 700 + 800
print (x)

