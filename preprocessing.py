### Preprocessing ###

## Step 0: Import relevant libraries ##
import numpy as np # for linear algebra
import pandas as pd # for data processing 

## Step 1: Read in the data ##
train_transaction_df = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/data/train_transaction.csv")
train_identity_df = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/data/train_identity.csv")

train_transaction_df.head()
train_transaction_df.shape # (590540, 394)
train_identity_df.head()
train_identity_df.shape # (144233, 41)

# Read in the data (test) 
test_transaction_df = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/data/test_transaction.csv")
test_identity_df = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/data/test_identity.csv")


## Step 2: Merge data ##
# "Not all transactions have corresponding identity information". Hence, we left join train_identity_df to train_transaction_df 
train_df = train_transaction_df.merge(train_identity_df, on = 'TransactionID', how = 'left')
train_df.shape # (590540, 434)
# Merge data (test)
test_df = test_transaction_df.merge(test_identity_df, on = 'TransactionID', how = 'left')


## Step 3: Memory Reduction ##
train_df.memory_usage().sum() # 2055079200 ~> 2GB of RAM required for training set parsing which is too much considering this is just the train set 

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

# By using mem_reduce, we have taken off approx 1.37GB of RAM for train data
train_mr_df = mem_reduce(train_df).copy()
test_mr_df = mem_reduce(test_df).copy()


## Step 4: Missing Values + Label Encoding ##
# Missing Values (for missing values we just replace the value with the string "missing")

# obtaining categorical column list cat_cols (train set)
cols_train = train_mr_df.columns 
num_cols_train = train_mr_df._get_numeric_data().columns
cat_cols_train = list(set(cols_train) - set(num_cols_train))
# obtaining categorical column list cat_cols (test set)
cols_test = test_mr_df.columns 
num_cols_test = test_mr_df._get_numeric_data().columns
cat_cols_test = list(set(cols_test) - set(num_cols_test))

for col in cat_cols_train:
    train_mr_df[col] = train_mr_df[col].fillna("missing")
# Missing Values (test)
for col in cat_cols_test:
    test_mr_df[col] = test_mr_df[col].fillna("missing")
    
# Label Encoding for categorical values (train)
for col in cat_cols_train:
    train_mr_df[col] = train_mr_df[col].astype("category")
    train_mr_df[col] = train_mr_df[col].cat.codes
# Label Encoding for categorical values (test) 
for col in cat_cols_test:
    test_mr_df[col] = test_mr_df[col].astype("category")
    test_mr_df[col] = test_mr_df[col].cat.codes

# imputing missing values in numeric columns to medians (train)
a = train_mr_df[num_cols_train].isnull().any()
train_null_num_cols = a[a].index
for n in train_null_num_cols:
    train_mr_df[f'{n}_isna'] = train_mr_df[n].isnull()
    median = train_mr_df[n].median()
    train_mr_df[n].fillna(median, inplace=True)
# imputing missing values in numeric columns to medians (test)
a = test_mr_df[num_cols_test].isnull().any()
test_null_num_cols = a[a].index
for n in test_null_num_cols:
    test_mr_df[f'{n}_isna'] = test_mr_df[n].isnull()
    median = test_mr_df[n].median()
    test_mr_df[n].fillna(median, inplace=True)

# checking if there are any null values in train and test sets after cleaning 
train_mr_df.isnull().values.any()
test_mr_df.isnull().values.any()


## Step 5: Exporting pre-processed train_df and test_df to csv ##
train_mr_df.to_csv('train_reduced_memory.csv')
test_mr_df.to_csv('test_reduced_memory.csv')