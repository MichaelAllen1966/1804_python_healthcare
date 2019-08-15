import queue

# FIFO (First In First Out) queue

print ('FIFO (First In First Out) Queue')

q = queue.Queue()

# Add items to queue (add numbers 0 to 4)
for item in range(5):
    q.put(item) 

# Retrieve items from queue using a loop. 
# See additional method below of continually retrieving
# until queue is empty.
    
for item in range(5):
    x = q.get() # put item in queue
    print (x, end = ' ')
    
# LIFO (Last In First Out) queue
    
print ('\nLIFO (Last In First Out) Queue')

q = queue.LifoQueue()

for item in range(5):
    q.put(item) # put item in queue
    
# Alternative way of emptying queue

while not q.empty():
    x = q.get() # put item in queue
    print (x, end = ' ')

# Priority queue

import random
print ('\nPriority Queue')

q = queue.PriorityQueue()

for i in range(5):
    priority = random.randint(1,100)
    name = 'Item ' + str(i)
    item = (priority, name) # A tuple is used to pass priority and item
    q.put(item) 
    
while not q.empty():
    x = q.get()
    print (x, end = ' ')