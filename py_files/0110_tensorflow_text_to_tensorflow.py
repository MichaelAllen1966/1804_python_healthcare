"""
This example starts with with raw text (movie reviews) and predicts whether the 
reviewer liked the movie.

The code goes through the following steps:
    1. import libraries
    2. load data
    3. clean data
    4. convert words to numbers
    5. process data for tensorflow
    6. build model
    7. train model
    8. predict outcome (like movie or nor) for previously unseen reviews

For information on installing a tensorflow environment in Anaconda see:
https://pythonhealthcare.org/2018/12/19/106-installing-and-using-tensorflow-using-anaconda/

For installing anaconda see:
https://www.anaconda.com/download

We import necessary libraries.

If you are missing a library then if using Anaconda from a command line (after
activating tensorflow library) use:
    conda import library-name
        
If you find you missing a nltk download then from a command line (after
activating tensorflow library) use:
    python (to being command line python session)
    import nltk
    nltk.download(library name)
    or
    nltk.download() will open dialoge box where you can install any/all nltk
    libraries
"""

###############################################################################
############################## IMPORT LIBRARIES ############################### 
###############################################################################

import numpy as np
import pandas as pd
import nltk
import tensorflow as tf

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from tensorflow import keras


# If not previously performed:
# nltk.download('stopwords')

###############################################################################
################################## LOAD DATA ################################## 
###############################################################################

"""
Here we load up a csv file. Each line contains a text string and then a label.
An example is given to download the imdb dataset which contains 50,000 movie
reviews. The label is 0 or 1 depending on whetehr the reviewer liked the movie.
"""
print ('Loading data')

# If you do not already have the data locally you may download (and save) by:
# file_location = 'https://gitlab.com/michaelallen1966/00_python_snippets' +\
#     '_and_recipes/raw/master/machine_learning/data/IMDb.csv'
# data = pd.read_csv(file_location)
# save to current directory
# data.to_csv('imdb.csv', index=False)

# If you already have the data locally then you may run the following
data = pd.read_csv('imdb.csv')

# Change headings of dataframe to make them more universal
data.columns=['text','label']

# We'll now hold back 5% of data for a final test that ha snot been used in
# training

number_of_records = data.shape[0]
number_to_hold_back = int(number_of_records * 0.05)
number_to_use = number_of_records - number_to_hold_back
data = data.head(number_to_use)
data_held_back = data.tail(number_to_hold_back)

###############################################################################
################################## CLEAN DATA ################################# 
###############################################################################

"""
Here we process the data in the following ways:
  1) change all text to lower case
  2) tokenize (breaks text down into a list of words)
  3) remove punctuation and non-word text
  4) find word stems (e.g. runnign, run and runner will be converted to run)
  5) removes stop words (commonly occuring words of little value, e.g. 'the')
"""

stemming = PorterStemmer()
stops = set(stopwords.words("english"))

def apply_cleaning_function_to_list(X):
    cleaned_X = []
    for element in X:
        cleaned_X.append(clean_text(element))
    return cleaned_X

def clean_text(raw_text):
    """This function works on a raw text string, and:
        1) changes to lower case
        2) tokenizes (breaks text down into a list of words)
        3) removes punctuation and non-word text
        4) finds word stems
        5) removes stop words
        6) rejoins meaningful stem words"""
    
    # Convert to lower case
    text = raw_text.lower()
    
    # Tokenize
    tokens = nltk.word_tokenize(text)
    
    # Keep only words (removes punctuation + numbers)
    # use .isalnum to keep also numbers
    token_words = [w for w in tokens if w.isalpha()]
    
    # Stemming
    stemmed_words = [stemming.stem(w) for w in token_words]
    
    # Remove stop words
    meaningful_words = [w for w in stemmed_words if not w in stops]
      
    # Return cleaned data
    return meaningful_words

print ('Cleaning text')
# Get text to clean
text_to_clean = list(data['text'])

# Clean text and add to data
data['cleaned_text'] = apply_cleaning_function_to_list(text_to_clean)

###############################################################################
######################## CONVERT WORDS TO NUMBERS ############################# 
###############################################################################

"""
The frequency of all words is counted. Words are then given an index number so
that th emost commonly occurring words hav ethe lowest number (so the 
dictionary may then be truncated at any point to keep the most common words).
We avoid using the index number zero as we will use that later to 'pad' out
short text.
"""

def training_text_to_numbers(text, cutoff_for_rare_words = 1):
    """Function to convert text to numbers. Text must be tokenzied so that
    test is presented as a list of words. The index number for a word
    is based on its frequency (words occuring more often have a lower index).
    If a word does not occur as many times as cutoff_for_rare_words,
    then it is given a word index of zero. All rare words will be zero.
    """

    # Flatten list if sublists are present
    if len(text) > 1:
        flat_text = [item for sublist in text for item in sublist]
    else:
        flat_text = text
    
    # get word freuqncy
    fdist = nltk.FreqDist(flat_text)

    # Convert to Pandas dataframe
    df_fdist = pd.DataFrame.from_dict(fdist, orient='index')
    df_fdist.columns = ['Frequency']

    # Sort by word frequency
    df_fdist.sort_values(by=['Frequency'], ascending=False, inplace=True)

    # Add word index
    number_of_words = df_fdist.shape[0]
    df_fdist['word_index'] = list(np.arange(number_of_words)+1)
    
    # Convert pandas to dictionary
    word_dict = df_fdist['word_index'].to_dict()
    
    # Use dictionary to convert words in text to numbers
    text_numbers = []
    for string in text:
        string_numbers = [word_dict[word] for word in string]
        text_numbers.append(string_numbers)
    
    return (text_numbers, df_fdist)

# Call function to convert training text to numbers
print ('Convert text to numbers')
numbered_text, dict_df = \
    training_text_to_numbers(data['cleaned_text'].values)

# Keep only word freqeuncies 1 to 10000
def limit_word_count(numbered_text):
    max_word_count = 10000
    filtered_text = []
    for number_list in numbered_text:
        filtered_line = \
            [number for number in number_list if number <=max_word_count]
        filtered_text.append(filtered_line)
        
    return filtered_text
    
data['numbered_text'] = limit_word_count(numbered_text)

# Pickle dataframe and dictionary dataframe (for later use if required)
data.to_pickle('data_numbered.p')
dict_df.to_pickle('data_dictionary_dataframe.p')

###############################################################################
######################### PROCESS DATA FOR TENSORFLOW ######################### 
###############################################################################

"""
Here we extract data from the pandas DataFrame, make all text vectors the same
length (by padding short texts and truncating long ones). We then split into
trainign and test data sets.
"""

print ('Process data for TensorFlow model')

# At this point pickled data (processed in an earlier run) might be loaded with 
# data=pd.read_pickle(file_name)
# dict_df=pd.read_pickle(filename)

# Get data from datframe and put in X and y lists
X = list(data.numbered_text.values)
y = data.label.values

## MAKE ALL X DATA THE SAME LENGTH
# We will use keras to make all X data a length of 512.
# Shorter data will be padded with 0, longer data will be truncated.
# We have oreviously kept the value zero free from use..

processed_X = \
    keras.preprocessing.sequence.pad_sequences(X,
                                               value=0,
                                               padding='post',
                                               maxlen=512)

## SPLIT DATA INTO TRAINIGN AND TEST SETS

X_train, X_test, y_train, y_test=train_test_split(
        processed_X,y,test_size=0.2,random_state=999)

###############################################################################
########################## BUILD MODEL AND OPTIMIZER ########################## 
###############################################################################

"""
Here we construct a four-layer neural network with keras/tensorflow.
The first layer is the input layer, then we have two hidden layers, and an
output layer.
"""

print ('Build model')

# BUILD MODEL

# input shape is the vocabulary count used for the text-to-numebr conversion 
# (10,000 words plus one for our zero padding)
vocab_size = 10001

###############################################################################
# Info on neural network layers
#
# The layers are stacked sequentially to build the classifier:
#
# The first layer is an Embedding layer. This layer takes the integer-encoded 
# vocabulary and looks up the embedding vector for each word-index. These 
# vectors are learned as the model trains. The vectors add a dimension to the 
# output array. The resulting dimensions are: (batch, sequence, embedding).
#
# Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for
# each example by averaging over the sequence dimension. This allows the model 
# to handle input of variable length, in the simplest way possible.
#
# This fixed-length output vector is piped through a fully-connected (Dense) 
# layer with 16 hidden units.
#
# The last layer is densely connected with a single output node. Using the 
# sigmoid activation function, this value is a float between 0 and 1, 
# representing a probability, or confidence level.
#
# The regulaizers help prevent over-fitting. Overfitting is evident when the
# trainign data fit is significant better than the test data fit. The level and
# the type may be adjusted to maximise test accuracy
##############################################################################

model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16))        

model.add(keras.layers.GlobalAveragePooling1D())

model.add(keras.layers.Dense(16, activation=tf.nn.relu, 
                             kernel_regularizer=keras.regularizers.l2(0.01)))

model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid,
                             kernel_regularizer=keras.regularizers.l2(0.01)))

model.summary()

# CONFIGURE OPTIMIZER

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

###############################################################################
################################# TRAIN MODEL ################################# 
###############################################################################

"""
Here we train the model. Using more epochs may give higher accuracy.

In 'real life' you may wish to hold back other test data (e.g. 10% of the 
orginal data so that you may use the test set here to help optimise the 
neutral network parameters and then test the final model on an independent data
set.

When verbose is set to 1, the model will show accuracy and loss for training 
and test data sets

"""

print ('Train model')

# Train model (verbose = 1 shows training progress)
model.fit(X_train,
          y_train,
          epochs=100,
          batch_size=512,
          validation_data=(X_test, y_test),
          verbose=1)


results = model.evaluate(X_train, y_train)
print('\nTraining accuracy:', results[1])

results = model.evaluate(X_test, y_test)
print('\nTest accuracy:', results[1])

###############################################################################
######################### PREDICT RESULTS FOR NEW TEXT ######################## 
###############################################################################

"""
Here we make predictions from text that has never been applied before. As we
are using data that has been held back we may also check its accuracy against 
a known label
 """

print ('\nMake predictions')

# We held some data back from thr original test set
# We will first clean the text

text_to_clean = list(data_held_back['text'].values)
X_clean = apply_cleaning_function_to_list(text_to_clean)
 
# Now we need to convert words to numbers.
# As these are new data it is possible that the word is not recognized so we
# will check the word is in the dictionary

# Convert pandas dataframe to dictionary
word_dict = dict_df['word_index'].to_dict()

# Use dictionary to convert words in text to numbers
text_numbers = []
for string in X_clean:
    string_numbers = []
    for word in string:
        if word in word_dict:
            string_numbers.append(word_dict[word])
    text_numbers.append(string_numbers)

# Keep only the top 10,000 words
# The function is repeated here for clarity (but would not usually be repeated)  

def limit_word_count(numbered_text):
    max_word_count = 10000
    filtered_text = []
    for number_list in numbered_text:
        filtered_line = \
            [number for number in number_list if number <=max_word_count]
        filtered_text.append(filtered_line)
        
    return filtered_text
    
text_numbers = limit_word_count(text_numbers)

# Process into fixed length arrays
    
processed_X = \
    keras.preprocessing.sequence.pad_sequences(text_numbers,
                                               value=0,
                                               padding='post',
                                               maxlen=512)

# Get prediction
predicted_classes = model.predict_classes(processed_X)
# The predicted classes give 0/1 for each possible class. As we only have one
# class we need to 'flatten' this array to remove nesting
predicted_classes = predicted_classes.flatten()

# Check prediction against known label
actual_classes = data_held_back['label'].values
accurate_prediction = predicted_classes == actual_classes
accuracy = accurate_prediction.mean()
print ('Accuracy on unseen data: %.2f' %accuracy)