from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC

from Utility import getData, printMetrics, getMetrics, logAndSave

splitData = False
if splitData:
	X_train, X_test, y_train, y_test = getData(useImbalancer=True, useStratify=True)
else:
	X_train, y_train = getData(splitData=splitData, useImbalancer=True, useStratify=True)
	X_test, y_test = None, None


def AdaBoostModel(splitData, X_train, X_test, y_train, y_test):
	svc = SVC()
	clf = AdaBoostClassifier(base_estimator=svc, algorithm='SAMME')
	grid_values = {'base_estimator__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],'base_estimator__C': [x / 10 for x in range(1, 11)],'base_estimator__degree': list(range(3, 5))}
	grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring=['roc_auc', 'f1', 'accuracy'], refit='roc_auc')
	grid_clf_acc.fit(X_train, y_train.ravel())
	clf = grid_clf_acc.best_estimator_

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
	# print("acc-" + str(acc) + "\tprecision-" + str(pre) + "\trecall-" + str(recall) + "\tauc-" + str(auc) + "\tf1-" + str(f1) + "\tval_accuracy-" + str(val_acc) + "\tval_precision-" + str(val_pre) + "\tval_recall-" + str(val_recall) + "\tval_auc-" + str(val_auc) + "\tval_f1-" + str(val_f1) + "\n")

	logAndSave(name_of_model="AdaBoostGS", clf=clf, metrics=metrics, val_metrics=val_metrics)


if __name__ == "__main__":
	AdaBoostModel(splitData, X_train, X_test, y_train, y_test)
