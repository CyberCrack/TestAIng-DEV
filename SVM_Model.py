from sklearn import svm

from Utility import getData, printMetrics, getMetrics, logAndSave, logAndSaveV2, getAnnealingData

splitData = False
if splitData:
	X_train, X_test, y_train, y_test = getData(useImbalancer=True, useStratify=False)
else:
	X_train, y_train = getData(splitData=splitData, useImbalancer=False, useStratify=False)
	X_test, y_test = None, None

X_train, X_test, y_train, y_test = getAnnealingData()


def SVMModel(splitData, X_train, X_test, y_train, y_test):
	clf = svm.SVC()
	clf.fit(X_train, y_train.ravel())

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

	logAndSave(name_of_model="SVM", clf=clf, metrics=metrics, val_metrics=val_metrics)


def SVMModelV2(X_train, X_test, y_train, y_test):
	multi_class = True
	clf = svm.SVC()
	clf.fit(X_train, y_train)

	y_preds = clf.predict(X_test)
	# printMetrics(y_test, y_preds, multi_class=multi_class)
	val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds, multi_class=multi_class)

	y_preds = clf.predict(X_train)
	# printMetrics(y_train, y_preds, multi_class=multi_class)
	acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds, multi_class=multi_class)
	val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
	metrics = (acc, pre, recall, auc, f1)
	# print("acc-" + str(acc) + "\tprecision-" + str(pre) + "\trecall-" + str(recall) + "\tauc-" + str(auc) + "\tval_accuracy-" + str(val_acc) + "\tval_precision-" + str(val_pre) + "\tval_recall-" + str(val_recall) + "\tval_auc-" + str(val_auc) + "\n")

	logAndSaveV2(name_of_model="SVMModelV2", clf=clf, metrics=metrics, val_metrics=val_metrics)


if __name__ == "__main__":
	# SVMModel(splitData, X_train, X_test, y_train, y_test)
	SVMModelV2(X_train, X_test, y_train, y_test)
