### Sampling ###

## Step 0: Importing relevant libraries ## 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.metrics import confusion_matrix
import numpy as np 


## Step 1: Reading in train_reduced_memory ##
train = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/train_reduced_memory.csv")
Y_train = train.iloc[:, 2].copy()
X_train = train.iloc[:, train.columns != 'isFraud'].copy()
print(Counter(Y_train)) # Counter({0: 569877, 1: 20663}) --> pre sampling

# produce a chart of isFraud distribution before sampling
xlabs = ['0','1']
ylabs = [Counter(Y_train)[0], Counter(Y_train)[1]]

plt.bar(xlabs, ylabs)
plt.title('Non-Fraud vs Fraud')
plt.xlabel('isFraud')
plt.ylabel('Observations')
plt.show()


## Step 2: SMOTE + Random Under Sampling ##
# define pipeline
oversample = SMOTE(sampling_strategy = 0.1)
undersample = RandomUnderSampler(sampling_strategy = 0.5)
steps = [('o', oversample), ('u', undersample)]
pipeline = Pipeline(steps=steps)

X_train1, Y_train1 = pipeline.fit_resample(X_train, Y_train)


#ran=RandomUnderSampler() 
#X_train1,Y_train1 = ran.fit_resample(X_train,Y_train)
print(Counter(Y_train1)) # Counter({0: 20663, 1: 20663}) --> post sampling

# produce a chart of isFraud distribution now after sampling 
xlabs = ['0','1']
ylabs = [Counter(Y_train1)[0], Counter(Y_train1)[1]]

plt.bar(xlabs, ylabs)
plt.title('Non-Fraud vs Fraud (RandomUnderSampling)')
plt.xlabel('isFraud')
plt.ylabel('Observations')
plt.ylim(0,569877)
plt.show()

## Step 3: Feature Selection ##
# V1~V339
cols_v = ['V' + str(x) for x in range(1,340)]
X_train_V = X_train1[cols_v].copy()
corr = X_train_V.corr()
sns.heatmap(corr)
# --- feature selection algo taken from https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf --- #
import numpy as np 
columns = np.full((corr.shape[0],), True, dtype = bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns_V = X_train_V.columns[columns]
X_train_V = X_train_V[selected_columns_V] # removed 151 cols (188 V cols remaining)

# D1 ~ D15
cols_d = ['D' + str(x) for x in range(1,16)]
X_train_D = X_train1[cols_d].copy()
corr = X_train_D.corr()
sns.heatmap(corr)
# --- feature selection algo taken from https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf --- #
import numpy as np 
columns = np.full((corr.shape[0],), True, dtype = bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns_D = X_train_D.columns[columns]
X_train_D = X_train_D[selected_columns_D] # removed 1 cols (14 D cols remaining)

# C1 ~ C14
cols_c = ['C' + str(x) for x in range(1,15)]
X_train_C = X_train1[cols_c].copy()
corr = X_train_C.corr()
sns.heatmap(corr)
# --- feature selection algo taken from https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf --- #
import numpy as np 
columns = np.full((corr.shape[0],), True, dtype = bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns_C = X_train_C.columns[columns]
X_train_C = X_train_C[selected_columns_C] # removed 11 cols (4 C cols remaining)

# M1 ~ M9
cols_m = ['M' + str(x) for x in range(1,10)]
X_train_M = X_train1[cols_m].copy()
corr = X_train_M.corr()
sns.heatmap(corr)
# --- feature selection algo taken from https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf --- #
import numpy as np 
columns = np.full((corr.shape[0],), True, dtype = bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns_M = X_train_M.columns[columns]
X_train_M = X_train_M[selected_columns_M] # removed 4 cols (5 M cols remaining)

# id_01 ~ id_38
cols_id = ['id_0' + str(x) for x in range(1,10)]
cols_idd = ['id_' + str(x) for x in range(10,39)]
cols_id = cols_id + cols_idd
X_train_ID = X_train1[cols_id].copy()
corr = X_train_ID.corr()
sns.heatmap(corr)
# --- feature selection algo taken from https://towardsdatascience.com/feature-selection-correlation-and-p-value-da8921bfb3cf --- #
import numpy as np 
columns = np.full((corr.shape[0],), True, dtype = bool)
for i in range(corr.shape[0]):
    for j in range(i+1, corr.shape[0]):
        if corr.iloc[i,j] >= 0.9:
            if columns[j]:
                columns[j] = False
selected_columns_ID = X_train_ID.columns[columns]
X_train_ID = X_train_ID[selected_columns_ID] # removed 7 cols (31 ID cols remaining)

# Reduced list of features 
cols_reduced = ['TransactionID', 'TransactionDT', 'TransactionAmt',
                'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
                'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain']

cols_reduced = cols_reduced + list(selected_columns_C) + list(selected_columns_D) + list(selected_columns_M) + list(selected_columns_V) + list(selected_columns_ID)

# Reduced features training set 
X_train_feature_reduce = X_train1[cols_reduced].copy()

# Reduced testing set (NEED TO REMOVE SAME FEATURES FOR TESTING SET)
test = pd.read_csv("/Users/lexy/Desktop/2021 UNSW/21T2/COMP9417/Final Project/test_reduced_memory.csv")
# need to replace id cols in test set to match train set ids (change id-01 to id_01)
test = test.rename(columns={ 'id-01': 'id_01', 'id-02': 'id_02', 'id-03': 'id_03', 'id-04': 'id_04','id-05': 'id_05','id-06': 'id_06','id-07': 'id_07',
'id-08': 'id_08','id-09': 'id_09','id-10': 'id_10','id-11': 'id_11','id-12': 'id_12','id-13': 'id_13','id-14': 'id_14','id-15': 'id_15',
'id-16': 'id_16','id-17': 'id_17','id-18': 'id_18','id-19': 'id_19','id-20': 'id_20','id-21': 'id_21','id-22': 'id_22','id-23': 'id_23',
'id-24': 'id_24','id-25': 'id_25','id-26': 'id_26','id-27': 'id_27','id-28': 'id_28','id-29': 'id_29','id-30': 'id_30','id-31': 'id_31',
'id-32': 'id_32','id-33': 'id_33','id-34': 'id_34','id-35': 'id_35', 'id-36': 'id_36','id-37': 'id_37','id-38': 'id_38'})

# then we can run the same command as above (on line 140) but for the test set 
X_test_feature_reduce = test[cols_reduced].copy()
Y_test_feature_reduce = test.iloc[:, test.columns == 'isFraud']


## Step 4: Training + Validation Split (80-20 split) ##
X_train2, X_val, Y_train2, Y_val = train_test_split(X_train_feature_reduce, Y_train1, test_size=0.33, random_state=42).copy()

## Step 5: Exporting Data ## 
X_train2.to_csv('X_train_final.csv')
Y_train2.to_csv('Y_train_final.csv')
X_val.to_csv('X_val_final.csv')
Y_val.to_csv('Y_val_final.csv')
X_test_feature_reduce.to_csv('X_test_final.csv')
Y_test_feature_reduce.to_csv('Y_test_final.csv')