
# coding: utf-8

# # Logistic regression
# 
# ## Load data
# 
# This example uses a data set built into sklearn. It classifies biopsy samples from breast cancer patients as ether malignant (cancer) or benign (no cancer).

# In[2]:


from sklearn import datasets

data_set = datasets.load_breast_cancer()

# Set up features (X), labels (y) and feature names
X = data_set.data
y = data_set.target
feature_names = data_set.feature_names
label_names = data_set.target_names


# Show feature names.

# In[3]:


print (feature_names)


# Show first record feature data.

# In[4]:


print (X[1])


# Show label names.

# In[5]:


print (label_names)


# Print first 25 lables.

# In[6]:


print (y[:25])


# We are dealing with a binary outcome. There are just two possibilities: benign or malignant. The methods described below will also work with problems with more than two possible classifications, but we'll keep things relatively simple here.

# ## Split the data into training and test sets
# 
# Data will be split randomly with 75% of the data used to train the machine learning model, and 25% used to test the model.

# In[7]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.25)


# ## Scale the data using standardisation
# 
# We scale data so that all features share similar scales. 
# 
# The X data will be transformed by standardisation. To standardise we subtract the mean and divide by the standard deviation. All data (training + test) will be standardised using the mean and standard deviation of the training set.
# 
# We will use a scaler from sklearn (but we could do this manually).

# In[8]:


from sklearn.preprocessing import StandardScaler

# Initialise a new scaling object for normalising input data
sc=StandardScaler() 

# Set up the scaler just on the training set
sc.fit(X_train)

# Apply the scaler to the training and test sets
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)


# Look at the first row of the raw and scaled data.

# In[9]:


print ('Raw data:')
print (X_train[0])
print ()
print ('Scaled data')
print (X_train_std[0])


# ## Fit logistic regression model
# 
# Our first machien learning model is a logistic regression model.
# 
# https://en.wikipedia.org/wiki/Logistic_regression

# In[10]:


from sklearn.linear_model import LogisticRegression

ml = LogisticRegression(C=1000)
ml.fit(X_train_std,y_train)


# That’s it! We can now use the model to predict malignant vs. benign classification of patients.

# In[11]:


# Predict training and test set labels
y_pred_train = ml.predict(X_train_std)
y_pred_test = ml.predict(X_test_std)


# ## Check accuracy of model

# In[12]:


import numpy as np

accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)

print ('Accuracy of prediciting training data =', accuracy_train)
print ('Accuracy of prediciting test data =', accuracy_test)


# Notice that the accuracy of fitting training data is significantly higher than the test data. This is known as over-fitting. There are two potential problems with over-fitting:
# 
# 1) If we test accuracy on the same data used to train the model we report a spuriously high accuracy. Our model is not actually as good as we report.
# 2) The model is too tightly built around our training data.
# 
# The solution to the first problem is to always report accuracy based on predicting the values of a test data set that was not used to train the model.
# 
# The solution to the second problem is to use 'regularisation'.

# ## Regularisation
# 
# Regularisation 'loosens' the fit to the training data. It effectively moves all predictions a little closer to the average for all values. 
# 
# In our logistic regression model, the regularisation parameter is C. A C value of 1,000 means there is very little regularisation. Try changing the values down by factors of 10, re-run code blocks 9, 10 and 11 and see what happens to the accuracy of the model. What value of C do you think is best?

# ## Examinining the probability of outcome, and changing the sensitivity of the model to predicting a positive
# 
# There may be cases where either:
# 
# 1) We want to see the probability of a given classification, or
# 
# 2) We want to adjust the sensitivity of predicting a certain outcome (e.g. for health screening we may choose to accept more false positives in order to minimise the number fo false negatives).
# 
# For linear regression we use the output 'predict_proba'. This may also be used in other machine learning models such as random forests and neural networks (but for support vector machines the output 'decision_function' is used in place of predict_proba.
# 
# Let's look at it in action.
# 
# For this section we'll refit the model with regularisation of C=0.1.

# In[13]:


# We'll start by retraining the model with C=0.1

ml = LogisticRegression(C=0.1)
ml.fit(X_train_std,y_train)
y_pred_test = ml.predict(X_test_std)

# Calculate the predicted probability of outcome 1:

y_pred_probability = ml.predict_proba(X_test_std)

# Print first 5 values and the predicted label:

print ('Predicted label probabilities:')
print (y_pred_probability[0:5])
print ()
print ('Predicted labels:')
print (y_pred_test[0:5])
print ()
print ('Actual labels:')
print (y_test[0:5])


# Let's calculate false positive and false negatives . In this data set being clear of chance has a label '1', and having cancer has a label '0'. A false positive, that is a sample is classed as cancer when is not actually cancer has a predicted test label of 0 and an actual label of 0. A false negative (predicted no cancer when cancer is actually present) has a predicted label of 1 and and actual label of zero.

# In[14]:


fp = np.sum((y_pred_test == 1) & (y_test == 0))
fn = np.sum((y_pred_test == 0) & (y_test == 1))

print ('False positives:', fp)
print ('False negatives:', fn)


# Maybe we are more concerned about false negatives. Let's adjust the probability cut-off to change the threshold for classification as having no cancer (predicted label 1).

# In[15]:


cutoff = 0.75

# Now let's make a prediction based on that new cutoff.
# Column 1 contains the probability of no cancer

new_prediction = y_pred_probability[:,1] >= cutoff


# And let's calculate our false positives and negatives:

# In[16]:


fp = np.sum((new_prediction == 0) & (y_test == 1))
fn = np.sum((new_prediction == 1) & (y_test == 0))

print ('False positives:', fp)
print ('False negatives:', fn)


# We have eliminated false negatives, but at the cost of more false postives. Try adjusting the cuttoff value. What value do you think is best?

# ## Model weights (coefficients)
# 
# We can obtain the model weights (coefficients) for each of the features. Values that are more strongly positive or negative are most important. A positive number means that this feature is linked to a classification label of 1. A negative number means that this feature is linked to a classification label of 0. 
# We can obtain the weights by using the method .coef_ (be careful to add the trailing underscore).

# In[17]:


print (ml.coef_)


# # Random Forests
# 
# A second type of categorisation model is a Random Forest.
# 
# https://en.wikipedia.org/wiki/Random_forests

# In[18]:


from sklearn.ensemble import RandomForestClassifier

ml = RandomForestClassifier(n_estimators = 10000,
                            n_jobs = -1)

# For random forests we don't need to use scaled data
ml.fit (X_train,y_train)


# In[19]:


# Predict test set labels

y_pred_test = ml.predict(X_test)
accuracy_test = np.mean(y_pred_test == y_test)
print ('Accuracy of prediciting test data =', accuracy_test)


# ## Feature importance
# 
# Feature importances give us the relative importance of each feature - the higher the number the greater the influence on the decision (they add up to 1.0). Feature importances do not tell use which decision is more likely.
# 
# (Careful to add the trailing underscore in ml.feature_importances_)

# In[20]:


import pandas as pd

df = pd.DataFrame()
df['feature'] = feature_names
df['importance'] = ml.feature_importances_
df = df.sort_values('importance', ascending = False)
print (df)


# # ADDITIONAL MATERIAL

# # Support Vector Machines
# 
# https://en.wikipedia.org/wiki/Support_vector_machine
# 
# Support Vector Machines are another common classification algorithm.
# Regularisation (C) may be adjusted, and different 'kernels' may also be applied. The two most common are 'linear' and 'rbf').

# In[21]:


# Import data

from sklearn import datasets

data_set = datasets.load_breast_cancer()

# Set up features (X), labels (y) and feature names

X = data_set.data
y = data_set.target
feature_names = data_set.feature_names
label_names = data_set.target_names

# Split data into training and test sets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(
        X,y,test_size=0.25)

# Scale data

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

# Fit model
# Note: a common test is to see whether linear or rbf kernel is best
# Try changing regularisation (C)

from sklearn.svm import SVC
ml = SVC(kernel='linear',C=100)
ml.fit(X_train_std,y_train)

# Predict training and test set labels

y_pred_train = ml.predict(X_train_std)
y_pred_test = ml.predict(X_test_std)

# Check accuracy of model

import numpy as np
accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)
print ('Accuracy of prediciting training data =', accuracy_train)
print ('Accuracy of prediciting test data =', accuracy_test)

# Show classification probabilities for first five samples
# Note that for SVMs we use decision_function, in place of predict_proba 

y_pred_probability = ml.decision_function(X_test_std)
print ()
print ('Predicted label probabilities:')
print (y_pred_probability[0:5])


# # Neural Networks
# 
# Neural networks may be better for very large or complex data sets. A challenge is the number of parameters that need to be optimised.
# 
# After importing the MLPClassifier (another name for a Neural Network is a Mulit-Level Perceptron Classifier) type help (MLPClassifier) for more information. 

# In[22]:


# Import data

from sklearn import datasets

data_set = datasets.load_breast_cancer()

# Set up features (X), labels (y) and feature names

X = data_set.data
y = data_set.target
feature_names = data_set.feature_names
label_names = data_set.target_names

# Split data into training and test sets

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)

# Scale data

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
sc.fit(X_train)
X_train_std=sc.transform(X_train)
X_test_std=sc.transform(X_test)

# Fit model
# Note: a common test is to see whether linear or rbf kernel is best
# Try changing regularisation (C)

from sklearn.neural_network import MLPClassifier

ml = MLPClassifier(solver='lbfgs',
                   alpha=1e-8, 
                   hidden_layer_sizes=(50, 10),
                   max_iter=100000, 
                   shuffle=True, 
                   learning_rate_init=0.001,
                   activation='relu', 
                   learning_rate='constant', 
                   tol=1e-7,
                   random_state=0)

ml.fit(X_train_std, y_train) 

# Predict training and test set labels

y_pred_train = ml.predict(X_train_std)
y_pred_test = ml.predict(X_test_std)

# Check accuracy of model

import numpy as np
accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)
print ('Accuracy of prediciting training data =', accuracy_train)
print ('Accuracy of prediciting test data =', accuracy_test)

# Show classification probabilities for first five samples
# Neural networks may often produce spuriously high proabilities!

y_pred_probability = ml.predict_proba(X_test_std)
print ()
print ('Predicted label probabilities:')
print (y_pred_probability[0:5])


# # A Random Forest example with multiple categories
# 
# We will use another classic 'toy' data set to look at multiple label classification. This is the categorisation of iris plants. We only have four features but have three different classification possibilities.
# We will use logistic regression, but all the methods described above work on multiple label classification. 
# Note: for completeness of code we'll import the required modules again, but this is not actually necessary if they have been imported previously.
# 
# ## Load data

# In[23]:


from sklearn import datasets

data_set = datasets.load_iris()

X = data_set.data
y = data_set.target
feature_names = data_set.feature_names
label_names = data_set.target_names

print ('Label names:')
print (label_names)
print ()
print ('Feature names:')
print (feature_names)
print ()
print ('First sample feature values:')
print (X[0])
print ()
print ('Labels:')
print (y)


# ## Split data into training and test data sets

# In[24]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25)


# ## Scale data
# 
# Here we'll use a random forest model which does not need scaling. But for all other model types a scaling step would be added here.

# ## Fit random forest model and show accuracy 

# In[25]:


ml = RandomForestClassifier(n_estimators = 10000,
                            random_state = 0,
                            n_jobs = -1)

# For random forests we don't need to use scaled data
ml.fit (X_train,y_train)


# ## Predict training and test set labels

# In[26]:


y_pred_train = ml.predict(X_train)
y_pred_test = ml.predict(X_test)


# ## Check accuracy of model

# In[27]:


import numpy as np

accuracy_train = np.mean(y_pred_train == y_train)
accuracy_test = np.mean(y_pred_test == y_test)

print ('Accuracy of prediciting training data =', accuracy_train)
print ('Accuracy of prediciting test data =', accuracy_test)


# ## Show classification of first 10 samples

# In[28]:


print ('Actual label:')
print (y_test[0:10])
print ()
print ('Predicted label:')
print (y_pred_test[0:10])


# ## Showing prediction probabilities and changing sensitivity to classification
# 
# As with a binary classification we may be interested in obtaining the probability of label classification, either to get an indicator of the certainty of classification, or to bias classification towards or against particular classes.
# 
# Changing sensitivity towards particular class labels is more complicated with multi-class problems, but the principle is the same as with a binary classification. We can access the calculated probabilities of classification for each label. Usually the one with the highest probability  is taken, but rules could be defined to bias decisions more towards one class if that is beneficial.
# 
# Here we will just look at the probability outcomes for each class. The usual rule for prediction is to simply take the one that is highest.

# In[29]:


y_pred_probability = ml.predict_proba(X_test)

print (y_pred_probability[0:5])

