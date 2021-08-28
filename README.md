# Credit_Fraud_Classification 
This is a readme.md file that provides information on contents of files below:
- preprocessing.py
- sampling.py
- randomForest.py

## preprocessing.py
The goal of this section is to ensure that we do the minimum required pre-processing for our data.

### Step 0: Import Libraries
We start off by importing relevant libraries numpy for linear algebra and pandas for data processing. 

### Step 1: Read Data
Read in the train and test data (train_transactions, train_identity, test_transactions, test_identity)

### Step 2: Merge Data
Combine the transaction and identity data by TransactionID doing a left join onto transaction. We do the same for the test data 

### Step 3: Memory Reduction
This is an important step as we can see that the training data alone requires around 2GB of RAM for a single parsing which is too much. Thus, we need to reduce the memory usage by downcasting features in the train and test data accordingly. The function mem_reduce() solves this issue for numeric features by downcasting where possible. 

### Step 4: Missing Values + Label Encoding
For categorical features, we first replace all Nan values with 'missing' and then proceed to do label encoding such that each category is mapped to a number. This is done as Random Forest requires numeric values in order to work.\
We also need to consider the missing values in the numeric columns in which we will replace these values with the median.\
We repeat for the test data set too. 

### Step 5: Exporting
We export the train (train_reduced_memory) and test (test_reduced_memory) datasets which are now memory reduced and cleaned so that we don't need to go through this process again. 


## sampling.py 
The goal of this section is to solve the data imbalance by undersampling our data. Additionally, we implement feature selection techniques to reduce the dimensionality of our data. 

### Step 0: Import Libraries 
Importing RandomUnderSampler, SMOTE, Pipeline, train_test_split, pyplot, Counter, Pandas, matplotlib, sns, warnings, confusion_matrix, numpy, seaborn. 

### Step 1: Read Data 
Read in train_reduced_memory which is obtained from the preprocessing file. We now split the data into X_train and Y_train to separate the features from the dependent variable. By doing some exploratory data analysis, we can see that there is a large data imbalance. 

### Step 2: Re-sampling 
using randomUnderSampler to reduce the samples from the majority set and balance out the data in combination with SMOTE to oversample the minority set. 

### Step 3: Feature Selection 
Creating correlation matrices for groups of input features in order to remove redundant features. Note that the features removed for the training set also need to be removed for the testing set (thus we have a small section dedicated to the testing section at the bottom of this step)

### Step 4: Training Validation Split
We can now split the data into training vs validation by taking 33% of the training set as the validation set. 

### Step 5: Exporing 
We export X_train_final, Y_train_final, X_val_final, Y_val_final, test_final to csv for further use. 


## randomForest.py 
This section is aimed at building our actual machine learning model (random forest) and tune our model for the best performance. 

### Step 0: Import libraries 
importing pandas, numpy, RandomForestClassifier, roc_auc_score, 

### Step 1: Read Data
Reading in X_train_final, Y_train_final, X_val_final, Y_val_final

### Step 2: Memory Reduction 
We do a final memory reduction procedure in order to save some processing ability for our modelling 

## Step 3: Random Forest Implementation 
Initial random forest model prediction using default hyperparameters provides an accuracy of #0.9999209027631302

## Step 4: Hyperparameter tuning 
Tuning for a subset of hyperparameters in order to improve performance further 

## Step 5: Final Model 
Final model obtained given specific hyperparameters. 
