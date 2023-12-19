import autokeras as ak

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from dataset_operations import *

# Load dataset
#DATA_PATH = '../datasets/VeReMi_Extension/mixalldata_clean.csv' 
#df = load_dataset_to_dataframe(DATA_PATH)
#
#
## Pre-process data
#corr_matrix = df.corr()
#remove_nan_features_corr(df,corr_matrix)
#
#col = 'class'
#
#X = df.loc[:,df.columns != col]
#y = df.loc[:,df.columns == col]
#
## Split dataset
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#
## Traing model
#clf = ak.StructuredDataClassifier()
#clf.fit(X_train,y_train,epochs=10)
#
## Test model
#y_pred = clf.predict(X_test)
#
#accuracy = accuracy_score(y_test,y_pred)
#
#print('Accuracy on Test: ', accuracy)

