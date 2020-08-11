from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC

from Utility import printMetrics, getMetrics, logAndSave, logAndSaveV2, getAnnealingData, getData

splitData = False
if splitData:
	X_train, X_test, y_train, y_test = getData(useImbalancer=True, useStratify=True)
else:
	X_train, y_train = getData(splitData=splitData, useImbalancer=True, useStratify=True)
	X_test, y_test = None, None

X_train, X_test, y_train, y_test = getAnnealingData()


def AdaBoostModel(splitData, X_train, X_test, y_train, y_test):
	svc = SVC()
	clf = AdaBoostClassifier(base_estimator=svc, n_estimators=100, algorithm='SAMME')
	clf.fit(X_train, y_train.ravel())

	if splitData:
		y_preds = clf.predict(X_test)
		printMetrics(y_test, y_preds)
		val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds)
	else:
		val_acc, val_pre, val_recall, val_auc, val_f1 = 0, 0, 0, 0, 0

	y_preds = clf.predict(X_train).reshape(-1, 1)
	acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds)
	val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
	metrics = (acc, pre, recall, auc, f1)
	# print("acc-" + str(acc) + "\tprecision-" + str(pre) + "\trecall-" + str(recall) + "\tauc-" + str(auc) + "\tval_accuracy-" + str(val_acc) + "\tval_precision-" + str(val_pre) + "\tval_recall-" + str(val_recall) + "\tval_auc-" + str(val_auc) + "\n")

	logAndSave(name_of_model="AdaBoost", clf=clf, metrics=metrics, val_metrics=val_metrics)


def AdaBoostModelV2(X_train, X_test, y_train, y_test):
	multi_class = True
	clf = AdaBoostClassifier(base_estimator=RandomForestClassifier(), n_estimators=200, algorithm='SAMME')
	clf.fit(X_train, y_train)

	y_preds = clf.predict(X_test)
	printMetrics(y_test, y_preds, multi_class=multi_class)
	val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds, multi_class=multi_class)

	y_preds = clf.predict(X_train)
	printMetrics(y_train, y_preds, multi_class=multi_class)
	acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds, multi_class=multi_class)
	val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
	metrics = (acc, pre, recall, auc, f1)
	# print("acc-" + str(acc) + "\tprecision-" + str(pre) + "\trecall-" + str(recall) + "\tauc-" + str(auc) + "\tval_accuracy-" + str(val_acc) + "\tval_precision-" + str(val_pre) + "\tval_recall-" + str(val_recall) + "\tval_auc-" + str(val_auc) + "\n")

	logAndSaveV2(name_of_model="AdaBoostModelV2", clf=clf, metrics=metrics, val_metrics=val_metrics)


if __name__ == "__main__":
	# AdaBoostModel(splitData, X_train, X_test, y_train, y_test)
	AdaBoostModelV2(X_train, X_test, y_train, y_test)
