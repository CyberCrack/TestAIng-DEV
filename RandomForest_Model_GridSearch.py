from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from Utility import getData, printMetrics, getMetrics, logAndSave, getAnnealingData, logAndSaveV2

splitData = True
if splitData:
	X_train, X_test, y_train, y_test = getData(useImbalancer=True, useStratify=True)
else:
	X_train, y_train = getData(splitData=splitData, useImbalancer=False, useStratify=True)
	X_test, y_test = None, None

X_train, X_test, y_train, y_test = getAnnealingData()


def RandomForestModel(splitData, X_train, X_test, y_train, y_test):
	clf = RandomForestClassifier(n_estimators=100, max_depth=11)
	grid_values = {'n_estimators': list(range(100, 501, 50)), 'criterion': ['gini', 'entropy'], 'max_depth': list(range(10, 21, 1))}
	grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring=['roc_auc', 'f1', 'accuracy'], refit='roc_auc')
	grid_clf_acc.fit(X_train, y_train.ravel())
	clf = grid_clf_acc.best_estimator_
	if splitData:
		y_preds = clf.predict(X_test)
		printMetrics(y_test, y_preds)
		val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds)
	else:
		val_acc, val_pre, val_recall, val_auc, val_f1 = 0, 0, 0, 0, 0
	y_preds = clf.predict(X_train)
	acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds)
	val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
	metrics = (acc, pre, recall, auc, f1)
	# print("acc-" + str(acc) + "\tprecision-" + str(pre) + "\trecall-" + str(recall) + "\tauc-" + str(auc) + "\tf1-" + str(f1) + "\tval_accuracy-" + str(val_acc) + "\tval_precision-" + str(val_pre) + "\tval_recall-" + str(val_recall) + "\tval_auc-" + str(val_auc) + "\tval_f1-" + str(val_f1) + "\n")

	logAndSave(name_of_model="RandomForestClassifierGS", clf=clf, metrics=metrics, val_metrics=val_metrics)


def RandomForestModelV2(X_train, X_test, y_train, y_test):
	multi_class = True
	clf = RandomForestClassifier()
	grid_values = {'n_estimators': list(range(100, 501, 50)), 'criterion': ['gini', 'entropy'], 'max_depth': list(range(10, 21, 1))}
	grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring=['roc_auc_ovr_weighted', 'f1_weighted', 'accuracy'], refit='f1_weighted', n_jobs=2, verbose=0)
	grid_clf_acc.fit(X_train, y_train)
	clf = grid_clf_acc.best_estimator_
	# print(clf)
	y_preds = clf.predict(X_test)
	# printMetrics(y_test, y_preds, multi_class=multi_class)
	val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds, multi_class=multi_class)

	y_preds = clf.predict(X_train)
	# printMetrics(y_train, y_preds, multi_class=multi_class)
	acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds, multi_class=multi_class)
	val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
	metrics = (acc, pre, recall, auc, f1)

	logAndSaveV2(name_of_model="RandomForestModelV2GS", clf=clf, metrics=metrics, val_metrics=val_metrics)


if __name__ == "__main__":
	# RandomForestModel(splitData, X_train, X_test, y_train, y_test)
	RandomForestModelV2(X_train, X_test, y_train, y_test)
