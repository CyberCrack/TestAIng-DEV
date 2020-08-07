import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from tpot import TPOTClassifier

from Utility import printMetrics

data = pd.read_csv(filepath_or_buffer="DataSource/binary.csv")
X = data.values[:, 1:-1]
rank_dummy = pd.get_dummies(data['rank'], drop_first=True).to_numpy()
X = np.concatenate((X, rank_dummy), axis=1)
y = data.values[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, shuffle=True, stratify=y)
standard_scaler = preprocessing.StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)
tpot = TPOTClassifier(generations=100, population_size=100, scoring='roc_auc', verbosity=2, random_state=101, cv=15, n_jobs=15)

tpot.fit(X_train, y_train)

print(tpot.score(X_test, y_test))
y_pred = tpot.predict(X_test)

printMetrics(y_test, y_pred)
y_pred = tpot.predict(X_train)
print("*"*100)
printMetrics(y_train, y_pred)
tpot.export('tpot_classifier_pipeline.py',data_file_path='DataSource/binary.csv')
