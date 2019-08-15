"""
When considering optimisation of multiple objectives, the Pareto front is that
collection of points where one objective cannot be improved without detriment to
another objective. These points are also called 'non-dominated'. In contrast,
points not on the Pareto front, or 'dominated' points represents points where it
is possible to improve one or more objectives without loss of performance of 
another objective.

Here we present code to identify points on the Pareto front.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# First we'll create some dummy data
# Each item has two scores

scores = np.array([[97, 23],
                  [55, 77],
                  [34, 76],
                  [80, 60],
                  [99,  4],
                  [81,  5],
                  [ 5, 81],
                  [30, 79],
                  [15, 80],
                  [70, 65],
                  [90, 40],
                  [40, 30],
                  [30, 40],
                  [20, 60],
                  [60, 50],
                  [20, 20],
                  [30,  1],
                  [60, 40],
                  [70, 25],
                  [44, 62],
                  [55, 55],
                  [55, 10],
                  [15, 45],
                  [83, 22],
                  [76, 46],
                  [56, 32],
                  [45, 55],
                  [10, 70],
                  [10, 30],
                  [79, 50]])



# Let's plot the scores

x = scores[:, 0]
y = scores[:, 1]

plt.scatter(x, y)
plt.xlabel('Objective A')
plt.ylabel('Objective B')
plt.show()

# FUNCTION TO IDENTIFY PARETO FRONT


# Now we will define our function to identify those points that are on the 
# Pareto front. We start off by assuming all points are on the Pareto front and
# then change the status of those that are not on the Pareto front. We use two 
# loops. The outer loop ('i') will loop through all points in order to compare
# them to all other points (the comparison is made using the inner loop, 'j').
# For any given point 'i', if any other point is at least as good in all 
# objectives and is better in one, that point 'i' is known as 'dominated' and is
# not on the Pareto front. As soon as a better point is found another (point is 
# at least as good in all objectives and is better in one), point i is marked as
# not on the Pareto front and the inner loop can stop.



def identify_pareto(scores):
    # Count number of items
    population_size = scores.shape[0]
    # Create a NumPy index for scores on the pareto front (zero indexed)
    population_ids = np.arange(population_size)
    # Create a starting list of items on the Pareto front
    # All items start off as being labelled as on the Parteo front
    pareto_front = np.ones(population_size, dtype=bool)
    # Loop through each item. This will then be compared with all other items
    for i in range(population_size):
        # Loop through all other items
        for j in range(population_size):
            # Check if our 'i' pint is dominated by out 'j' point
            if all(scores[j] >= scores[i]) and any(scores[j] > scores[i]):
                # j dominates i. Label 'i' point as not on Pareto front
                pareto_front[i] = 0
                # Stop further comparisons with 'i' (no more comparisons needed)
                break
    # Return ids of scenarios on pareto front
    return population_ids[pareto_front]


# We'll now apply our function to our data, and print the 
pareto = identify_pareto(scores)
print ('Pareto front index vales')
print ('Points on Pareto front: \n',pareto)

pareto_front = scores[pareto]
print ('\nPareto front scores')
print (pareto_front)

# Use Pandas to sort values on the Pareto front (only needed for plotting)
pareto_front_df = pd.DataFrame(pareto_front)
pareto_front_df.sort_values(0, inplace=True)
pareto_front = pareto_front_df.values

# Select parteo front points
# Plot pareto front on out chart

x_all = scores[:, 0]
y_all = scores[:, 1]
x_pareto = pareto_front[:, 0]
y_pareto = pareto_front[:, 1]

plt.scatter(x_all, y_all)
plt.plot(x_pareto, y_pareto, color='r')
plt.xlabel('Objective A')
plt.ylabel('Objective B')
plt.show()
