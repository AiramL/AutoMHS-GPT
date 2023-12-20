from sys import path
path.append('/root/anaconda3/pkgs/auto-sklearn-0.15.0-pyhd8ed1ab_0/site-packages')

import autosklearn.classification as ask

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from autosklearn.metrics import accuracy,f1,precision,recall

from dataset_operations import *

# Load dataset
DATA_PATH = '../datasets/VeReMi_Extension/mixalldata_clean.csv' 
df = load_dataset_to_dataframe(DATA_PATH)


# Pre-process data
corr_matrix = df.corr()
remove_nan_features_corr(df,corr_matrix)

col = 'class'

X = df.loc[:,df.columns != col]
y = df.loc[:,df.columns == col]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Traing model
clf = ask.AutoSklearnClassifier(
        time_left_for_this_task=600,
        max_models_on_disc=100,
        metric=accuracy,
        ensemble_size=3,
        scoring_functions=[accuracy,precision,f1,recall],
        memory_limit=1024000)
clf.fit(X_train,y_train)

# Test model
y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print('Accuracy on Test: ', accuracy)

