def add_three_numbers(a, b, c):
    return a + b + c

x=[10, 20, 35]


# To unpack a list or a tuple for use in a function expecting separate
# arguments, use an * before the list/tuple name
print(add_three_numbers(*x))