from sys import path
path.append('/root/miniconda3/lib/python3.11/site-packages/')

import h2o
from h2o.automl import H2OAutoML as hml

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataset_operations import *

# Load dataset
DATA_PATH = '../datasets/VeReMi_Extension/mixalldata_clean.csv' 
df = load_dataset_to_dataframe(DATA_PATH)


# Pre-process data
corr_matrix = df.corr()
remove_nan_features_corr(df,corr_matrix)

col = 'class'

X = df#.loc[:,df.columns != col]
#y = df.loc[:,df.columns == col]

# Split dataset
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
X_train, X_test = train_test_split(X,test_size=0.2,random_state=42)

## Traing model
h2o.init()

train = h2o.H2OFrame(X_train) 

x = train.columns
y = 'class'
x.remove(y)

aml = hml(max_models=20, seed=1)
aml.train(x=x,y=y,training_frame=train)

lb = aml.leaderboard
lb.head(rows=lb.nrows)

clf = aml.leader

save_data(clf,'../models/h2o_best')

#test = h2o.H2OFrame(X_test) 
#
### Test model
#y_pred = clf.predict(X_test)
#
#accuracy = accuracy_score(y_test,y_pred)
#
#print('Accuracy on Test: ', accuracy)

