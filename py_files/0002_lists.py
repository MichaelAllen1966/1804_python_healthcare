
# coding: utf-8

# # Lists
# 
# Lists, tuples, sets and dictionaries are four basic objects for storing data in Python. Later we'll look at libraries that allow for specialised handling and analysis of large or complex data sets.
# 
# Lists are a group of text or numbers (or more complex objects) held together in a given order. Lists are recognised by using square brackets , with members of the list separated by commas. Lists are mutable, that is their contents may be changed. Note that when Python numbers a list the first element is referenced by the index zero. Python is a 'zero-indexed' language.
# 
# Below is code demonstrating some common handling of lists.

# ## Creating a list
# Lists may mix text, numbers, or other Python objects. Here is code to generate and print a list of mixed text and numbers. Note that the text is in speech marks (single or double; both will work). If the speech marks were missing then Python would think the text represented a variable name (such as the name of another list).

# In[1]:


my_list = ['Gandalf','Bilbo','Gimli',10,30,40]
print (my_list)


# ## Deleting a list
# 
# The <em>del</em> command will delete a list (or any other variable).

# In[4]:


my_list = ['Gandalf','Bilbo','Gimli']
del my_list


# ## Returning an element from a list
# 
# Python list store their contents in sequence. A particular element may be referenced by its index number. Python indexes start at zero. Watch out for this - this will almost certainly trip you up at some point.
# 
# The example below prints the second item in the list. The first item would be referenced with 0 (zero).

# In[4]:


my_list = ['Gandalf','Bilbo','Gimli',10,30,40]
print (my_list[1])


# Negative references may be used to refer to position from the <em>end</em> of the list. An index of -1 would be the last item (because -0 is the same as 0 and would be the first item!).

# ## Taking 'slices' of a list
# 
# Taking more than one element from a list is called a slice. The slice defines the starting point, and the point just after the end of the slice. The example below takes elements indexed 2 to 4 (but not 5).

# In[1]:


my_list = [0,1,2,3,4,5,6,7,8,9,10]
slice = my_list[2:5]
print(slice)


# Slices can take every n elements, the example below takes every second element from index 1 up to, but not including, index 9 (remembering that the first element is index 0):
# 

# In[2]:


y_list = [0,1,2,3,4,5,6,7,8,9,10]
slice = my_list[2:9:2]
print(slice)


# Reverse slices may also be taken by defining the end point before the beginning point and using a negative step:

# In[3]:


my_list = [0,1,2,3,4,5,6,7,8,9,10]
slice = my_list[9:2:-2]
print(slice)


# Colons can be used to designate beginning or end. For example:

# In[3]:


my_list = [0,1,2,3,4,5,6,7,8,9,10]
print (my_list[:4])
print (my_list[3:])
print (my_list[::2])


# ## Deleting or changing an item in a list
# 
# Unlike tuples (which are very similar to lists), elements in a list may be deleted or changed.
# 
# Here the first element in the list is deleted:

# In[5]:


my_list = ['Gandalf','Bilbo','Gimli',10,30,40]
del my_list[0]
print (my_list)


# Here the second element of the list is replaced:

# In[6]:


my_list = ['Gandalf','Bilbo','Gimli',10,30,40]
my_list[0] = 'Boromir'
print (my_list)


# ## Appending an item to a list, and joining two lists
# 
# Individual elements may be added to a list with append.

# In[7]:


my_list = ['Gandalf','Bilbo','Gimli',10,30,40]
my_list.append ('Elrond')
print (my_list)


# Two lists may be joined with a simple +

# In[9]:


my_list = ['Gandalf','Bilbo','Gimli']
my_second_list = ['Elrond','Boromir']
my_combined_list = my_list + my_second_list
print (my_combined_list)


# Or the <em>extend</em> method may be used to add a list to an existing list.
# 

# In[11]:


my_list = ['Gandalf','Bilbo','Gimli']
my_second_list = ['Elrond','Boromir']
my_list.extend(my_second_list)
print (my_list)


# ## Copying a list
# 
# If we try to copy a list by saying 'my_copy = my_list' we do not generate a new copy. Instead we generate an 'alias', another name that refers to the original list. Any changes to my_copy will change my_list.
# 
# There are two ways we can make a separate copy of a list, where changes to the new copy will not change the original list:

# In[12]:


my_list = ['Gandalf','Bilbo','Gimli']
my_copy = my_list.copy()
# Or
my_copy = my_list[:]


# ## Inserting an element into a given position in a list
# 
# An element may be inserted into a list with the <em>insert</em> method. Here we specify that the new element will be inserted at index 2 (the third element in the list, remembering that the first element is index 0). In a later post we will look at methods specifically for maintaining a sorted list, but for now let us simply inset an element at a given position in the list.

# In[13]:


my_list = ['Gandalf','Bilbo','Gimli']
my_list.insert(2, 'Boromir')
print (my_list)


# ## Sorting a list
# 
# The <em>sort</em> method can work on a list of text or a list of numbers (but not a mixed list text and numbers). Sort order may be reversed if wanted.
# 
# A normal sort:

# In[14]:


x = [10, 5, 24, 5, 8]
x.sort()
print (x)


# A reverse sort:

# In[15]:


x = [10, 5, 24, 5, 8]
x.sort(reverse=True)
print (x)


# ## Checking whether the list contains a given element
# 
# The <em>in</em> command checks whether a particular element exist in a list. It returns a 'Boolean' True or False.
# 

# In[18]:


my_list = ['Gandalf','Bilbo','Gimli','Elrond','Boromir']
is_in_list = 'Elrond' in my_list
print (is_in_list)
is_in_list = 'Bob' in my_list
print (is_in_list)


# ## Reversing a list
# 
# The method to reverse a list looks a little odd. It is actually creating a 'slice' of a list that steps back from the end of list.

# In[1]:


my_list = ['Gandalf','Bilbo','Gimli','Elrond','Boromir']
my_list = my_list[::-1]
print (my_list)


# ## Counting the number of elements in a list
# 
# Use the <em>len</em> function to return the number of elements in a list.

# In[20]:


my_list = ['Gandalf','Bilbo','Gimli','Elrond','Boromir']
print (len(my_list))


# ## Returning the index position of an element of a list
# 
# The list <em>index</em> will return the position of the first element of the list matching the argument given.

# In[21]:


my_list = ['Gandalf','Bilbo','Gimli','Elrond','Boromir']
index_pos = my_list.index('Elrond')
print (index_pos)


# If the element is not in the list then the code will error. Later we will look at some different ways of coping with this type of error. For now let us introduce an if/then/else statement (and we'll cover those in more detail later as well).

# In[24]:


my_list = ['Gandalf','Bilbo','Gimli','Elrond','Boromir']
test_element = 'Elrond'
is_in_list = test_element in my_list
if is_in_list: # this is the same as saying if is_in_list == True
    print (test_element,'is in list at position', my_list.index(test_element))
else:
    print (test_element, 'is not in list')


# ## Removing an element of a list by its value
# 
# An element may be removed from a list by using its value rather than index. Be aware that you might want to put in a method (like the if statement above) to check the value is present before removing it. But here is a simple remove (this will remove the first instance of that particular value:

# In[25]:


my_list = ['Gandalf','Bilbo','Gimli','Elrond','Boromir']
my_list.remove('Bilbo')
print (my_list)


# ## 'Popping' an element from a list'
# 
# Popping an element from a list means to retrieve an element from a list, removing it from the list at the same time. If no element number is given then the last element of the list will be popped.

# In[26]:


my_list = ['Gandalf','Bilbo','Gimli','Elrond','Boromir']
print ('Original list:', my_list)
x = my_list.pop()
print (x, 'was popped')
print ('Remaining list:',my_list)
x = my_list.pop(1)
print (x, 'was popped')
print ('Remaining list:', my_list)


# ## Iterating (stepping) through a list
# 
# Lists are 'iterable', that is they can be stepped through. The example below prints one element at a time. We use the variable name 'x' here, but any name may be used.

# In[30]:


my_list = [1,2,3,4,5,6,7,8,9]
for x in my_list:
    print(x, end=' ') # end command replaces a newline with a space


# ## Iterating through a list and changing element values
# 
# Iterating through a list and changing the value in the original list is a little more complicated. Below the <em>range</em> command creates a range of numbers from zero up to the index of the last element of the list. If there were 4 elements, for example, then the range function would produce '0, 1, 2, 3'. This allows us to iterate through the index numbers for the list (the position of each element). Using the index number we can change the value of the element in that position of the list.

# In[32]:


my_list = [1,2,3,4,5,6,7,8,9]
for index in range(len(my_list)):
    my_list[index] = my_list[index]**2
print ('\nMy list after iterating and mutating:')
print (my_list)


# ## Counting the number of times an element exists in a list
# 
# The <em>count</em> method counts the number of times a value occurs in a list. If the value does not exist a zero is returned.

# In[33]:


x =[1, 2, 3, 4, 1, 4, 1]
test_value = 1
print ('Count the number of times 1 occurs in',x)
print (x.count(test_value),'\n')


# ## Turning a sentence into a list of words
# 
# The <em>split</em> method allows a long string, such as a sentence, to be separated into individual words. If no separator is defined any white space (such as a space or comma) will be used to divide words.

# In[34]:


my_sentence ="Turn this sentence into a list"
words = my_sentence.split()
print (words)


# A separator may be specified, such as diving sentences at commas:

# In[36]:


my_sentence = 'Hello there, my name is Bilbo'
divided_sentence = my_sentence.split(sep=',')
print(divided_sentence)

