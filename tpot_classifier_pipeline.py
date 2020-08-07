import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
from Utility import printMetrics, logAndSave, getMetrics

tpot_data = pd.read_csv('DataSource/binary.csv', sep=',', dtype=np.float64)
X = tpot_data.values[:, 1:-1]
rank_dummy = pd.get_dummies(tpot_data['rank'], drop_first=True).to_numpy()
X = np.concatenate((X, rank_dummy), axis=1)
y = tpot_data.values[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, shuffle=True, stratify=y)
standard_scaler = preprocessing.StandardScaler()
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_test)
# Average CV score on the training set was: 0.7358352229780801
exported_pipeline = make_pipeline(
	StackingEstimator(estimator=ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.9500000000000001, min_samples_leaf=9, min_samples_split=18, n_estimators=100)),
	StackingEstimator(estimator=MLPClassifier(alpha=0.001, learning_rate_init=1.0)),
	StackingEstimator(estimator=GradientBoostingClassifier(learning_rate=0.5, max_depth=4, max_features=0.6500000000000001, min_samples_leaf=9, min_samples_split=19, n_estimators=100, subsample=0.9500000000000001)),
	BernoulliNB(alpha=10.0, fit_prior=False)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 101)

exported_pipeline.fit(X_train, y_train)
y_preds = exported_pipeline.predict(X_train)
acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds)

y_preds = exported_pipeline.predict(X_test)
val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds)
val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)

metrics = (acc, pre, recall, auc, f1)
logAndSave(name_of_model="TPOT_Classifier", clf=exported_pipeline, metrics=metrics, val_metrics=val_metrics)
