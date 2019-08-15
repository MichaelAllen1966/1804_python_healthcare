
# If needed install seaborn (conda install seaborn or pip install seaborn)

import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers

###############################################################################
############################## LOAD DATA ######################################
###############################################################################

# Load data from web and save locally
dataset_path = keras.utils.get_file("auto-mpg.data", 
    "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
dataset_path
column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin'] 
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                      na_values = "?", comment='\t',
                      sep=" ", skipinitialspace=True)

raw_dataset.to_csv('mpg.csv', index=False)

# Load data locally
#data = pd.read_csv('mpg.csv')

###############################################################################
############################## CLEAN DATA #####################################
###############################################################################

# Dataset contains some missing data (see by using print(data.isna().sum()))
# Drop rows with missing data
data = data.dropna()

# The "Origin" column is really categorical, not numeric. 
# So convert that to a one-hot:

origin = data.pop('Origin')
data['USA'] = (origin == 1)*1.0
data['Europe'] = (origin == 2)*1.0
data['Japan'] = (origin == 3)*1.0

###############################################################################
############################## CLEAN DATA #####################################
###############################################################################

train_dataset = data.sample(frac=0.8,random_state=0)
test_dataset = data.drop(train_dataset.index)

###############################################################################
############################# EXAMINE DATA ####################################
###############################################################################

# Have a quick look at the joint distribution of a few pairs of columns from
# the training set.

g = sns.pairplot(
        train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], 
        diag_kind="kde")

fig = g.fig # convert to matplotlib plot. Other seaborn use fig = g.getfig()
fig.show()

# Look at overall stats
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()
print (train_stats)

###############################################################################
###################### SPLIT FEATURES FROM LABELS #############################
###############################################################################

train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

###############################################################################
########################### NORMALISE THE DATA ################################
###############################################################################

# Normalise using the mean and standard deviation from the training set

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

###############################################################################
############################### BUILD MODEL ###################################
###############################################################################

# Here, we'll use a Sequential model with two densely connected hidden layers,
# and an output layer that returns a single, continuous value. Regularisation
# helps prevent over-fitting (try adjusting the values; higher numbers = more
# regularisation. Regularisation may be type l1 or l2.)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01),
                 activation=tf.nn.relu, 
                 input_shape=[len(train_dataset.keys())]),
                                
    keras.layers.Dense(64, kernel_regularizer=keras.regularizers.l1(0.01),
                 activation=tf.nn.relu),
                       
    keras.layers.Dense(1)])

  optimizer = tf.train.RMSPropOptimizer(0.001)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model

model = build_model()

# Print a summary of the model

print (model.summary())

###############################################################################
############################### TRAIN MODEL ###################################
###############################################################################

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

# Show last few epochs in history
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

###############################################################################
############################### PLOT TRAINING #################################
###############################################################################

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [MPG]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,5])
  
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$MPG^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.legend()
  plt.ylim([0,20])
  plt.show()

plot_history(history)

###############################################################################
############################# MAKE PREDICTIONS ################################
###############################################################################

# Make predictions from test-set

test_predictions = model.predict(normed_test_data).flatten()

# Scatter plot plot
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# Error plot
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show

# Copyright (c) 2017 François Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

