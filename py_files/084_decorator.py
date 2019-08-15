"""
Example of simple function decorator
------------------------------------
@time_it is a function decorator. It takes the function below it and passes it,
along with the passed arguments, to the time_it function. This allows multiple
functions to be timed without replication of code, and keeps each function code
clean and uncluttered.
"""


def time_it(func):
    import time
    def wrapper (*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print (func.__name__ + ' took ' + str(int((end-start)*1000)) + 
               ' milliseconds')
        return result
    return wrapper

@time_it
def calc_square(numbers):
    result = []
    for number in numbers:
        result.append (number ** 2)
    return result


@time_it
def calc_cube(numbers):
    result = []
    for number in numbers:
        result.append (number ** 3)
    return result

if __name__ == '__main__':
    x = range(1,10000)
    calc_square(x)
    calc_cube(x)