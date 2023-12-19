
from sklearn.ensemble import RandomForestClassifier
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

X = df.loc[:,df.columns != col]
y = df.loc[:,df.columns == col]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

# Traing model
rf = RandomForestClassifier(random_state=42,n_estimators=150,
                            max_depth=None,min_samples_split=2,
                            min_samples_leaf=1)

rf = rf.fit(X_train,y_train)

# Test model
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)

print('Accuracy on Test: ', accuracy)

