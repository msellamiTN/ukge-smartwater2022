import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import pstdev
import scipy.stats as ss
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report,roc_auc_score, confusion_matrix, accuracy_score
import pickle
#import umap.umap_ as umap
import seaborn as sns
df_final= pd.read_csv("../data/water/FuzzyDataSetWater.csv", sep=";" ,encoding= 'unicode_escape')
# Calculating Water Quality Index of each sample
df_num_final = df_final.select_dtypes(exclude="object")
# Importing the dataset
df_new= pd.read_csv("../data/water/FuzzyDataSetWater.csv", sep=";" ,encoding= 'unicode_escape')
# Calculating Water Quality Index of each sample
Y = df_new['WQI clf']
df_new.drop('WQI clf', axis=1, inplace = True)
df_new.drop(['ID','STATION CODE', 'LOCATIONS', 'STATE','TempW','Total_ColW','Total_ColW.1','ConductivityW','NIW','DOW'], axis = 1, inplace = True)
#df_new.drop('ID', axis=1, inplace = True)
df_new.replace([np.inf, -np.inf], np.nan, inplace=True)
#Finding the mean of the column having NaN
mean_value=df_new['Conductivity'].mean()
  
df_new['Conductivity'].fillna(value=mean_value, inplace=True)
 
df_new.isnull().any() 

X_train, X_valid, Y_train, Y_valid = train_test_split(df_new, Y, test_size = 0.3)

print('------------SVM--------------\n')

y_true=Y_valid

clf_model = svm.SVC(kernel='linear', random_state = 0,probability=True).fit(X_train, Y_train)
#save the model
pickle.dump(clf_model, open('model.pkl', 'wb'))

