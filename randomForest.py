### RandomForest Implementation ###

## Step 0: Import relevant libraries ##
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RandomizedSearchCV

## Step 1: Read in the data ##
X_train_final = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/X_train_final.csv")
Y_train_final = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/Y_train_final.csv")
X_val_final = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/X_val_final.csv")
Y_val_final = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/Y_val_final.csv")

## Step 2: Memory Reduction ##
# --- memory reduction function inspired by https://www.kaggle.com/kabure/extensive-eda-and-modeling-xgb-hyperopt --- #
def mem_reduce(df):
    numeric_types = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64'] # all types of numerics 
    
    initial_memory_usage = df.memory_usage().sum()

    for col in df.columns: # looping through all columns in dataframe 
        col_type = df[col].dtypes # extracting the data type for col
 
        if col_type in numeric_types: # Checking if column is a numeric data type
            col_min = df[col].min() # min value of that col
            col_max = df[col].max() # max value of that col

            if str(col_type)[:3] == 'int': # int
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)  
            else: # float
                if col_min > np.finfo(np.float16).min and col_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64) 

    memory_usage = df.memory_usage().sum() 
    print("initial memory usage was:", initial_memory_usage)
    print("After memory reduction, memory is now:", memory_usage)
    print("The difference between the two is:", initial_memory_usage - memory_usage)
    return df 

# mem reduce X_train
X_train_final_mem_red = mem_reduce(X_train_final).copy()

# mem reduce Y_train
Y_train_final_mem_red = mem_reduce(Y_train_final).copy()
Y_train_final_mem_red = Y_train_final_mem_red.iloc[:, Y_train_final_mem_red.columns == 'isFraud']
Y_train_final_mem_red = Y_train_final_mem_red.iloc[:, 0]

# mem reduce X_val, Y_val
X_val_final_mem_red = mem_reduce(X_val_final).copy()
Y_val_final_mem_red = mem_reduce(Y_val_final).copy()
Y_val_final_mem_red = Y_val_final_mem_red.iloc[:, Y_val_final_mem_red.columns == 'isFraud']
Y_val_final_mem_red = Y_val_final_mem_red.iloc[:, 0]


## Step 3: Random Forest Implementation ##

#rfc = RandomForestClassifier(n_jobs = -1, verbose = 1, max_features = "sqrt") #0.9999472685087534 
rfc = RandomForestClassifier(n_jobs = -1, verbose = 1) #0.9999209027631302

print("rfc.fit")
rfc.fit(X_train_final_mem_red,Y_train_final_mem_red)

print("predictions")
y_pred = rfc.predict(X_val_final_mem_red)

print("roc_auc_score")
roc_auc_score(Y_val_final_mem_red, y_pred)

## Step 4: Hyperparameter tuning ## 
# --- hyperparameter tuning algo referenced here https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74 --- #
from sklearn.model_selection import RandomizedSearchCV
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['auto', 'sqrt', 'log2']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'max_features': max_features,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap,
               'n_estimators': n_estimators,
               'max_depth': max_depth,
               }

rf = RandomForestClassifier()

# Random search of parameters, using 3 fold cross validation, 
# search across 10 different combinations
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=42, n_jobs = -1)

# Fit the random search model
rf_random.fit(X_train_final_mem_red, Y_train_final_mem_red)

## Step 5: Best model using optimal hyperparameters ## 
rf_random.best_params_

rfc = RandomForestClassifier(n_jobs = -1, verbose = 1, n_estimators = 400, min_samples_split = 2, min_samples_leaf = 4, max_features = 'auto', max_depth = 80, bootstrap = False) #0.9999209027631302

print("rfc.fit")
rfc.fit(X_train_final_mem_red,Y_train_final_mem_red)

print("predictions")
y_pred = rfc.predict(X_val_final_mem_red)

print("roc_auc_score")
roc_auc_score(Y_val_final_mem_red, y_pred)

