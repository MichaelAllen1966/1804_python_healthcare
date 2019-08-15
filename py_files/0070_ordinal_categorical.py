
# coding: utf-8

# # Dealing with ordinal and categorical data
# 
# Some data sets may have ordinal data, which are descriptions with a natural order, such as small, medium large. There may also be categorical data which has no obvious order like green, blue, red. We'll usually want to convert both of these into numbers for use by machine learning models.
# 
# Let's look at an example:

# In[8]:


import pandas as pd

colour = ['green', 'green', 'red', 'blue', 'green', 'red','red']
size = ['small', 'small', 'large', 'medium', 'medium','x large', 'x small']

df = pd.DataFrame()
df['colour'] = colour
df['size'] = size

print (df)


# ## Working with ordinal data

# One of our columns is obviously ordinal data: size has a natural order to it. We can convert this text to a number by mapping  a dictionary to the column. We will create a new column (size_number) which replaces the text with a number.

# In[9]:


# Define mapping dictionary:

size_classes = {'x small': 1,
                'small': 2,
                'medium': 3,
                'large': 4,
                'x large': 5}

# Map to dataframe and put results in a new column:

df['size_number'] = df['size'].map(size_classes)

# Display th new dataframe:

print (df)


# ## Working with categorical data
# 
# There is no obvious sensible mapping of colour to a number. So in this case we create an extra column for each colour and put a one in the relevant column. For this we use pandas <em>get_dummies</em> method.    

# In[10]:


colours_df = pd.get_dummies(df['colour'])

print (colours_df)


# We then combine the new dataframe with the original one, and we can delete the temporary one we made:

# In[11]:


df = pd.concat([df, colours_df], axis=1, join='inner')

del colours_df

print (df)


# ## Selecting just our new columns
# 
# At the moment we have both the orginal data and the transformed data. For use in the model we would just keep the new columns. Here we'll use the pandas <em>loc</em> method to select column slices from size_number onwards:

# In[17]:


df1 = (df.loc[:,'size_number':])

print (df1)

