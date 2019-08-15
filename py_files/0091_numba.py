import numpy as np
import timeit
from numba import jit

import numpy as np
import timeit
from numba import jit

# Define a function normally without using numba

def test_without_numba():
    for i in np.arange(1000):
        x = i ** 0.5
        x *= 0.5
    
# Define a function using numba jit. Using the argument nopython=True gives the
# fastest possible run time, but will error if numba cannot precomplile all the
# code. Using just @jit will allow the code to mix pre-compiled and normal code
# but will not be as fast as possible

@jit(nopython=True)
def test_with_numba():
    for i in np.arange(1000):
        x = i ** 0.5
        x *= 0.5

# Run functions first time without timing (compilation included in first run)
test_without_numba()
test_with_numba()

# Time functions with timeit (100 repeats). 
# Multiply by 1000 to give milliseconds

timed = timeit.timeit(stmt = test_without_numba, number=100) * 1000
print ('Milliseconds without numba: %.3f' %timed)

timed = timeit.timeit(stmt = test_with_numba, number=100) * 1000
print ('Milliseconds with numba: %.3f' %timed)
