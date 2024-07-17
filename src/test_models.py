from pickle import load
from dataset_operations import *
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import autokeras as ak
import tensorflow as tf

from sys import path
path.append('/root/anaconda3/pkgs/auto-sklearn-0.15.0-pyhd8ed1ab_0/site-packages')

import pandas as pd 
import numpy as np
import seaborn as sns
import scipy.io as scio


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



AK_VE=True

# autogpt
#clf = read_obj('../models/autogpt_model')

# autokeras
#clf = tf.keras.models.load_model('structured_data_classifier/best_model',custom_objects=ak.CUSTOM_OBJECTS)

# auto-sklearn
clf = read_obj('../models/auto_sklearn_model_final')

# Load dataset
if AK_VE:
    DATA_PATH = '../datasets/VeReMi_Extension/mixalldata_clean.csv' 
    df = load_dataset_to_dataframe(DATA_PATH)
    # Pre-process data
    corr_matrix = df.corr()
    remove_nan_features_corr(df,corr_matrix)

    col = 'class'

    X = df.loc[:,df.columns != col]
    y = df.loc[:,df.columns == col]

else:
    dataset_1 = scio.loadmat('../datasets/WiSec_DataModifiedVeremi_Dataset/attack16withlabels.mat')
    dataset_2 = scio.loadmat('../datasets/WiSec_DataModifiedVeremi_Dataset/attack1withlabels.mat')
    dataset_3 = scio.loadmat('../datasets/WiSec_DataModifiedVeremi_Dataset/attack2withlabels.mat')
    dataset_4 = scio.loadmat('../datasets/WiSec_DataModifiedVeremi_Dataset/attack4withlabels.mat')
    dataset_5 = scio.loadmat('../datasets/WiSec_DataModifiedVeremi_Dataset/attack8withlabels.mat')

    header = ["type",
         "timeReceiver",
         "receiverID",
         "receiverXposition",
         "receiverYposition",
         "receiverZposition",
         "timeTransmitted",
         "senderID",
         "messageID",
         "senderXposition",
         "senderYposition",
         "senderZposition",
         "senderXvelocity",
         "senderYvelocity",
         "senderZvelocity",
         "rssi",
         "class"]

    df_dataset = pd.concat([pd.DataFrame(dataset_1['attack16withlabels']),
                 pd.DataFrame(dataset_2['attack1withlabels']),
                 pd.DataFrame(dataset_3['attack2withlabels']),
                 pd.DataFrame(dataset_4['attack4withlabels']),
                 pd.DataFrame(dataset_5['attack8withlabels'])])

    df_dataset.columns = header

    df_dataset = df_dataset.drop(['receiverID','senderID', 'messageID'], axis=1)

    df_dataset = df_dataset.dropna()

    features_nan_corr = ["receiverZposition",
                     "senderZposition",
                     "type",
                     "senderZvelocity",
                     "timeReceiver"]

    df_dataset = df_dataset.drop(columns=features_nan_corr)

    X = df_dataset.drop(columns=['class'])
    
    columns_names = X.columns
    scaler = MinMaxScaler()
    scaler = scaler.fit(X)
    X = pd.DataFrame(scaler.transform(X))
    X.columns = columns_names

    y = df_dataset['class']
    y = pd.get_dummies(y,columns=['class'])


results = ''

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)



# slice the test into 5 parts
partions = 5
for index in range(partions):
    
    if not AK_VE:
        y_pred_prob = clf.predict(X_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)])
        y_pred = prob_to_class(y_pred_prob)
        y_pred = pd.get_dummies(y_pred)
        results+=str((accuracy_score(y_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)],y_pred)))+','
        results+=str((recall_score(y_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)],y_pred,average='macro')))+','
        results+=str((precision_score(y_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)],y_pred,average='macro')))+','
        results+=str((f1_score(y_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)],y_pred,average='macro')))+'\n'

    else:
        y_pred_prob = clf.predict(X_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)])
        y_pred = prob_to_class(y_pred_prob)
        results+=str((accuracy_score(y_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)],y_pred)))+','
        results+=str((recall_score(y_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)],y_pred,average='macro')))+','
        results+=str((precision_score(y_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)],y_pred,average='macro')))+','
        results+=str((f1_score(y_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)],y_pred,average='macro')))+'\n'

#with open('ak_results','w') as writer:
#    writer.writelines(results)

print(results)
#print(f1_score(y_test[len(X_test)//partions*index:len(X_test)//partions*(index+1)],y_pred,average='samples'))
