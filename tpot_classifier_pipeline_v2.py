from sklearn.ensemble import ExtraTreesClassifier

from Utility import getAnnealingData, printMetrics, getMetrics, logAndSave

X_train, X_test, y_train, y_test = getAnnealingData()
multi_class = True

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
# tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
# features = tpot_data.drop('target', axis=1)
# training_features, testing_features, training_target, testing_target = \
#             train_test_split(features, tpot_data['target'], random_state=101)

# Average CV score on the training set was: 0.9886533912345425
exported_pipeline = ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.55, min_samples_leaf=1, min_samples_split=3, n_estimators=100)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
	setattr(exported_pipeline, 'random_state', 101)

exported_pipeline.fit(X_train, y_train)
y_preds = exported_pipeline.predict(X_test)
val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds, multi_class=multi_class)
val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
# printMetrics(y_test, y_preds, multi_class=multi_class)

y_preds = exported_pipeline.predict(X_train)
acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds, multi_class=multi_class)
metrics = (acc, pre, recall, auc, f1)
# printMetrics(y_train, y_preds, multi_class=multi_class)

logAndSave(name_of_model="TPOT_ClassifierV2", clf=exported_pipeline, metrics=metrics, val_metrics=val_metrics)
