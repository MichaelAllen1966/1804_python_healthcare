#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
List comprehension examples

List comprehensions are an alternative to for loops
They work specifically with Python Lists.

The code examples below give an introduction to using them. 

1. Double a list of numbers
2. Call a function from a list comprehension
3. Using zip within a list function to iterate multiple lists
4. Using If statments within a list comprehension
5. Creating a list of lists using a nested list comprehension
6. Looping through a list of lists using a list comprehension

@author: tom

"""

#%%
# =============================================================================
# Example 1 - double the numbers
# =============================================================================

foo = [1, 2, 3, 4]
bar = []

for x in foo:
    bar.append(x * 2)
    
print(bar)

#%%

# =============================================================================
# list comprehension approach for the same result...
# =============================================================================

foo = [1, 2, 3, 4]
bar = [x * 2 for x in foo]
print(bar)

#%%

# =============================================================================
# Example 2 - convert celsius to fahrenheit
# This example calls a function from within the list comprehension.
# =============================================================================

def convert_celsius_to_fahrenheit(deg_celsius):
    """
    Convert degress celsius to fahrenheit
    Returns float value - temp in fahrenheit
    Keyword arguments:
        def_celcius -- temp in degrees celsius
    """
    return (9/5) * deg_celsius + 32

#list of temps in degree celsius to convert to fahrenheit
celsius = [39.2, 36.5, 37.3, 41.0]

#standard for loop approach
fahrenheit = []
for x in celsius:
    fahrenheit.append(convert_celsius_to_fahrenheit(x))

        
print('using standard for loop: {}'.format(fahrenheit))

#implementation using a list comprehension
fahrenheit = [convert_celsius_to_fahrenheit(x) for x in celsius]
print('using list comprehension: {}'.format(fahrenheit))

#%%
# =============================================================================
# Example 3 - convert the strings to different data types
# This example also make ue of the zip function
# Zip allow you to iterate through two lists at the same time
# =============================================================================

inputs = ["1", "3.142", "True", "spam"]
converters = [int, float, bool, str]

values_with_correct_data_types = [t(s) for (s, t) in zip(inputs, converters)]
print(values_with_correct_data_types)

#%%
# =============================================================================
# Example 4 - Using if statements within a list comprehension
# The example filters a list of file names to the python files only
# =============================================================================

unfiltered_files = ['test.py', 'names.csv', 'fun_module.py', 'prog.config']

python_files = []

# filter the files using a standard for loop 
for file in unfiltered_files:
    if file[-2:] == 'py':
        python_files.append(file)
        
print('using standard for loop: {}'.format(python_files))

#list comprehension
python_files = [file for file in unfiltered_files if file[-2:] == 'py']

print('using list comprehension {}'.format(python_files))


#%%
# =============================================================================
# Example 5 - List comprehension to create a list of lists
# List comprehensions can greatly reduce the complexity of code
# needed to create a list of lists.
# =============================================================================

list_of_lists = []

for i in range(5):
    sub_list = []
    for j in range(3):
        sub_list.append(i * j)
    list_of_lists.append(sub_list)

print(list_of_lists)

#a lists comprehension reduces 6 lines of code to 1
list_of_lists = [[i * j for j in range(3)] for i in range(5)]

print(list_of_lists)


#%%
# =============================================================================
# Example 6: Iterate over all items in a list of lists
# using a list comprehension
# The code converts a list of lists to a list of items
# We call this flattening the list.
# =============================================================================

list_of_lists = [[8, 2, 1], [9, 1, 2], [4, 5, 100]]

flat_list = []
for row in list_of_lists:
    for col in row:
        flat_list.append(col)

print(flat_list)

#implementation as list comprehension
flat_list = [item for sublist in list_of_lists for item in sublist]

print(flat_list)


#%%
