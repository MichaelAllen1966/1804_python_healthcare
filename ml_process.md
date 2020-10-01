# Machine Learning process:

## Decide on basic model type

* Categorisation
* Regression problem
* Unsupervised (e.g. clustering)
* Natural Language Processing required

## Explore data

Use numerical (e.g. pandas) and graphical (e.g. matplotlib) to understand:

* Data types for each feature (numeric, ordinal categorical)
* Amount of missing data
* Censoring (capping) of data
* Correlations with output label
* Co-variance between features

## Create any key extra features (and remove old ones)

* Do any features need converting (e.g. time/data stamps to time intervals)
* Convert ordinal data (e.g. small/medium/large) to integers
* Convert categorical data (e.g. hospital/ward name) to 'one-hot' coding

## Clean data

* Deal with unusual data (e.g. remove sample, censor data to set min/max)
* Deal with missing data (e.g. remove sample, impute with median, etc.)
* Have columns to mark when data is imputed
* Highlight unusual rows?

## Consider whether synthetic data is needed (e.g. for highly unbalanced data)

* Use oversampling, undersampling, or SMOTE

## Split data 

* Split data in training and test sets (often 80:20)
* Test data is not used in model selection/optimisation
* Training data may be split again during optimisation (k-fold recommended)

## Model selection and optimisation

* The original test set is not used here
* This is performed using k-fold cross-validation (for training/test splits)
* Normalise data (e.g. standard scaling)
* Decide on basic performance measure: e.g. accuracy, f1, ROC, Brier score
* Perform base case models (with no significant optimisation)
    * Regression:
        * Multiple linear regression
        * Multiple polynomial regression
        * Random Forest regression
    * Classification
        * Logistic regression
        * Support Vector Machines (linear + RBF kernels)
        * Random Forests
        * Neural Networks
    * Clustering
        * k-means
        * Heirachial clustering
        * Topic modelling
    * NLP pre-processing
        * Tokenization 
        * Stemming
        * Stop word removal
        * Bag of words
        * POS (Parts of Speech_ tagging
        * Embedding (e.g. Word-2-Vec)
* Measure learning rates of base case models
* Consider feature engineering:
    * Reduce by feature selection or principle component analysis
    * Expand by polynomial features
    * Feature expansion may be followed by feature reduction
* Model hyper-parameter optimisation
    * Grid search
    * Randomised search
    * Heuristic (e.g. genetic algorithm?)
    * Statistical (e.g. statistical design of experiments)
    
## Model testing
    * Use held-back data
    * Decide on primary measure for comparison
        (e.g.accuracy, f1, ROC, Brier score)
    * Consider using boot-strapping to run multiple tests and measure sem
    
## Documentation and communication

### Global explainability - about the model (not individual cases):

* Empirical basis for algorithm, pedigree
* Representativeness of training set
* Can see working?
* What are, in general, the most influential items of information?
* Black box explainability - various methods (e.g. surrogates, layers)
* Results of evaluations

See checklists such as SUNDAE, ECONSORT

### Local explainability - about individual cases

* What drove the conclusion? e.g. LIME (local model)
* What of the inputs had been different (counterfactuals)
* What was the chain of reasoning?
* What tipped the balance?
* Is the current case within the model's competence?
* How confident is the conclusion?

### Example of output for users (for different audiences)

* Verbal gist
* Multiple graphical and numerical outputs, with instant 'what-ifs'
* Test and tables showing methods
* Mathematics
* Code

### Communicating uncertainty

* Model should communicate uncertainty/margin of error for each decision
* Communicating uncertainty is part of trustworthiness
* Output probabilities should be calibrated against accuracy (DS has published)

### Fairness

Consider possible sources of bias in the model.

* Model should be without bias and also perceived to be without bias
* Bias may be unintentional (e.g. use of postcode introduces racial bias)
