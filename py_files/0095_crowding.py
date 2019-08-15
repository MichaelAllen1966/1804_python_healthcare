import numpy as np
import random as rn

def calculate_crowding(scores):
    # Crowding is based on a vector for each individual
    # All scores are normalised between low and high. For any one score, all
    # solutions are sorted in order low to high. Crowding for chromsome x
    # for that score is the difference between the next highest and next
    # lowest score. Total crowding value sums all crowding for all scores

    population_size = len(scores[:, 0])
    number_of_scores = len(scores[0, :])

    # create crowding matrix of population (row) and score (column)
    crowding_matrix = np.zeros((population_size, number_of_scores))

    # normalise scores (ptp is max-min)
    normed_scores = (scores - scores.min(0)) / scores.ptp(0)

    # calculate crowding distance for each score in turn
    for col in range(number_of_scores):
        crowding = np.zeros(population_size)

        # end points have maximum crowding
        crowding[0] = 1
        crowding[population_size - 1] = 1

        # Sort each score (to calculate crowding between adjacent scores)
        sorted_scores = np.sort(normed_scores[:, col])

        sorted_scores_index = np.argsort(
            normed_scores[:, col])

        # Calculate crowding distance for each individual
        crowding[1:population_size - 1] = \
            (sorted_scores[2:population_size] -
             sorted_scores[0:population_size - 2])

        # resort to orginal order (two steps)
        re_sort_order = np.argsort(sorted_scores_index)
        sorted_crowding = crowding[re_sort_order]

        # Record crowding distances
        crowding_matrix[:, col] = sorted_crowding

    # Sum croding distances of each score
    crowding_distances = np.sum(crowding_matrix, axis=1)

    return crowding_distances

def reduce_by_crowding(scores, number_to_select):
    # This function selects a number of solutions based on tournament of
    # crowding distances. Two members of the population are picked at
    # random. The one with the higher croding dostance is always picked
    
    population_ids = np.arange(scores.shape[0])

    crowding_distances = calculate_crowding(scores)

    picked_population_ids = np.zeros((number_to_select))

    picked_scores = np.zeros((number_to_select, len(scores[0, :])))

    for i in range(number_to_select):

        population_size = population_ids.shape[0]

        fighter1ID = rn.randint(0, population_size - 1)

        fighter2ID = rn.randint(0, population_size - 1)

        # If fighter # 1 is better
        if crowding_distances[fighter1ID] >= crowding_distances[
            fighter2ID]:

            # add solution to picked solutions array
            picked_population_ids[i] = population_ids[
                fighter1ID]

            # Add score to picked scores array
            picked_scores[i, :] = scores[fighter1ID, :]

            # remove selected solution from available solutions
            population_ids = np.delete(population_ids, (fighter1ID),
                                       axis=0)

            scores = np.delete(scores, (fighter1ID), axis=0)

            crowding_distances = np.delete(crowding_distances, (fighter1ID),
                                           axis=0)
        else:
            picked_population_ids[i] = population_ids[fighter2ID]

            picked_scores[i, :] = scores[fighter2ID, :]

            population_ids = np.delete(population_ids, (fighter2ID), axis=0)

            scores = np.delete(scores, (fighter2ID), axis=0)

            crowding_distances = np.delete(
                crowding_distances, (fighter2ID), axis=0)

    # Convert to integer 
    picked_population_ids = np.asarray(picked_population_ids, dtype=int)
    
    return (picked_population_ids)



test_array = np.array([[12, 0],
                       [11.5, 0.5],
                       [11, 1],
                       [10.8, 1.2],
                       [10.5, 1.5],
                       [10.3, 1.8],
                       [9.5, 2],
                       [9, 2.5],
                       [7, 3],
                       [5, 4],
                       [2.5, 6],
                       [2, 10],
                       [1.5, 11],
                       [1, 11.5],
                       [0.8, 11.7],
                       [0, 12]])

crowding_distance = calculate_crowding(test_array)

print(crowding_distance)

selected_individuals = reduce_by_crowding(test_array, 5)

print (selected_individuals)

