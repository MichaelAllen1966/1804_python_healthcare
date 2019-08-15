
# coding: utf-8

# Here we look at the standard Python random number generator. It uses a <em>Mersenne Twister</em>, one of the mostly commonly-used random number generators. The generator can generate random integers, random sequences, and random numbers according to a number of different distributions. 

# ## Importing the random module and setting a random seed

# In[3]:


import random


# Random numbers can be repeatable in the sequence they are generated if a 'seed' is given. If no seed is given the random number sequence generated each time will be different.

# In[4]:


random.seed() # This will reset the random seed so that the random number sequence will be different each time
random.seed(10) # giving a number will lead to a repatable random number sequence each time


# ## Random numbers
Random integers between (and including) a minimum and maximum:
# In[5]:


print (random.randint(0,5))


# Return a random number between 0 and 1:

# In[6]:


print (random.random())


# Return a number (floating, not integer) between a & b:

# In[7]:


print (random.uniform(0,5))


# Select from normal distribution (with mu, sigma)

# In[8]:


print (random.normalvariate(10,3))


# Other distributions (see help(random) for more info after importing random module):
# 
# Lognormal, Exponential, Beta, Gaussian, Gamma, Weibul

# ## Generating random sequences
# 
# The random library may also be used to shuffle a list:

# In[11]:


deck = ['ace','two','three','four']
random.shuffle(deck)
print (deck)


# Sampling without replacement:

# In[13]:


sample = random.sample([10, 20, 30, 40, 50], k=4) # k is the number of samples to select
print (sample)


# Sampling with replacement:

# In[20]:


pick_from = ['red', 'black', 'green']
sample = random.choices(pick_from, k=10)
print(sample)


# Sampling with replacement and weighted sampling:

# In[19]:


pick_from = ['red', 'black', 'green']
pick_weights = [18, 18, 2]
sample = random.choices(pick_from, pick_weights, k=10)
print(sample)

