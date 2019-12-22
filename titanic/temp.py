import numpy as np

def calculate_accuracy(observed, predicted):
    
    """
    Calculates a range of acuuracy scores from observed and predicted classes,
    and returns a dictionary of results.
    
    Takes two list or NumPy arrays (observed class values, and predicted class 
    values).
    
     1) observed positive rate
     2) Predicted positive rate
     3) observed negative rate
     4) Predicted neagtive rate    
     5) accuracy - proportion of test results that are correct    
     6) precision: proportion of test positives that are real positives
     7) recall: proportion of true postives correctly identified
     8) f1: harmonic mean of precision and recall
     9) Sensitivity: Same as recall
    10) Specificity: Proportion of true negatives identified:        
    11) positive likelihood: increased probability of true +ve if test +ve
    12) negative likelihood: reduced probability of true +ve if test -ve
    13) false positive rate: proportion of false +ves in true -ve patients
    14) false negative rate: proportion of false -ves in true +ve patients
    15) positive predictive value: chance of true +ve if test +ve
    16) negative predictive value: chance of true -ve if test -ve

   
    """
    
    # Convert list to NumPy arrays
    if type(observed) == list:
        observed = np.array(observed)
    if type(predicted) == list:
        predicted = np.array(predicted)
    
    # Calculate accuracy scores
    observed_positives = observed == 1
    observed_negatives = observed == 0
    predicted_positives = predicted == 1
    predicted_negatives = predicted == 0
    
    true_positives = (predicted_positives == 1) & (observed_positives == 1)
    
    false_positives = (predicted_positives == 1) & (observed_positives == 0)
    
    true_negatives = (predicted_negatives == 1) & (observed_negatives == 1)
    
    accuracy = np.mean(predicted == observed)
    
    precision = (np.sum(true_positives) /
                 (np.sum(true_positives) + np.sum(false_positives)))
        
    recall = np.sum(true_positives) / np.sum(observed_positives)
    
    sensitivity = recall
    
    f1 = 2 * ((precision * recall) / (precision + recall))
    
    specificity = np.sum(true_negatives) / np.sum(observed_negatives)
    
    positive_likelihood = sensitivity / (1 - specificity)
    
    negative_likelihood = (1 - sensitivity) / specificity
    
    false_postive_rate = 1 - specificity
    
    false_negative_rate = 1 - sensitivity
    
    positive_predictive_value = (np.sum(true_positives) / 
                                 np.sum(observed_positives))
    
    negative_predicitive_value = (np.sum(true_negatives) / 
                                  np.sum(observed_positives))
    
    # Create dictionary for results, and add results
    results = dict()
    
    results['observed_positive_rate'] = np.mean(observed_positives)
    results['observed_negative_rate'] = np.mean(observed_negatives)
    results['predicted_positive_rate'] = np.mean(predicted_positives)
    results['predicted_negative_rate'] = np.mean(predicted_negatives)
    results['accuracy'] = accuracy
    results['precision'] = precision
    results['recall'] = recall
    results['f1'] = f1
    results['sensivity'] = sensitivity
    results['specificity'] = specificity
    results['positive_likelihood'] = positive_likelihood
    results['negative_likelihood'] = negative_likelihood
    results['false_postive_rate'] = false_postive_rate
    results['false_negative_rate'] = false_negative_rate
    results['positive_predictive_value'] = positive_predictive_value
    results['negative_predicitive_value'] = negative_predicitive_value
    
    return results


observed =  [0, 0, 1, 0, 1, 0, 1, 0, 1, 0]
predicted = [0, 0, 1, 0, 1, 0, 1, 0, 0, 1]

accuracy = calculate_accuracy(observed, predicted)

# Print resulys
for key, value in accuracy.items():
    # Print up to three decimal places
    print (key, "{0:0.3}".format(value))
