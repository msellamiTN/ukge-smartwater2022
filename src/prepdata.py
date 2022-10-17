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
df_new.drop(['ID','STATION CODE', 'LOCATIONS', 'STATE','TempW','ConductivityW','NIW','DOW'], axis = 1, inplace = True)
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

y_pred_svm = clf_model.predict(X_valid)
# Insert row to the dataframe using DataFrame.append()
data={}
uncertaingraph = pd.DataFrame(data)
for index in range(df_final.shape[0]):
        index_row = df_final.iloc[index, :]
        new_row = {'STATION':df_final['STATION CODE'][index], 'relation':'hastemp', 'temp': df_final['Temp'][index], 'confidence':df_final['TempW'][index]}
        uncertaingraph = uncertaingraph.append(new_row, ignore_index=True)
        new_row = {'STATION':df_final['STATION CODE'][index], 'relation':'hasDO', 'DO': df_final['DO'][index], 'confidence':df_final['DOW'][index]}
        uncertaingraph = uncertaingraph.append(new_row, ignore_index=True)
        new_row = {'STATION':df_final['STATION CODE'][index], 'relation':'hasConductivity', 'Conductivity': df_final['Conductivity'][index], 'confidence':df_final['ConductivityW'][index]}
        uncertaingraph = uncertaingraph.append(new_row, ignore_index=True)
        new_row = {'STATION':df_final['STATION CODE'][index], 'relation':'hasNI', 'NI': df_final['NI'][index], 'confidence':df_final['NIW'][index]}
        uncertaingraph = uncertaingraph.append(new_row, ignore_index=True)
        #Calcul de proabilité de la classe WQI C
        #Calcul de proabilité de la classe WQI C
        X_valid= index_row[['Temp' ,   'DO'   ,    'Conductivity' ,  'NI'   , 'Tot_col' ,    'WQI' ] ] 
        print(X_valid)
        X_features = np.asarray(X_valid).reshape(1, -1)  
         
        y_proba = clf_wz2.predict_proba(X_features,probability=True)
       
        new_row = {'STATION':df_final['STATION CODE'][index], 'relation':'hasWQIClass', 'WQI': df_final['WQI'][index], 'confidence':y_proba}
         
        uncertaingraph = uncertaingraph.append(new_row, ignore_index=True)
		
uncertaingraph.to_csv('../data/water/train.tsv', sep="\t",header=False,index=False)

