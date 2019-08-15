
# coding: utf-8

# # Nested lists
# 
# So far we have looked at lists which contain a simple series of numbers, text, or a mixture of numbers and texts (Python lists can also hold any Python object, but in Healthcare modelling we are usually dealing with numbers or text in a list).
# 
# It is possible though to build nested lists. In the example below we generate a list manually, with each nested list on a separate line. This separation by line is just to make it easier to see; it is not needed in Python, but thought to layout of code is important if other people will be looking at your code.

my_list = [[1,2,3],
          [4,5,6],
          [7,8,9]]
print (my_list)


# We can then refer to items by using two reference indices. The first refers to the nested list block (so the first id [1, 2, 3] and the second refers to the position within that block. Remember that Python is zero indexed so the first element of the first list is 0[0]

print (my_list[1][1])

# Or we can use other variables to refer to the position:

x = 2
y = 0
print (my_list[x][y])


# More complex structures can be built up with further nesting of lists to give multi-dimensional lists.
# 
# WARNING: Handling large arrays of data this way is possible but slow. For modelling we are much better off using two libraries dedicated to fast handling of large data sets: NumPy and Pandas. We will be covering those libraries soon.
