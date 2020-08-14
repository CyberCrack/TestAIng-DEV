from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from Utility import getData, printMetrics, getMetrics, logAndSave, logAndSaveV2, getAnnealingData

splitData = True
if splitData:
	X_train, X_test, y_train, y_test = getData(useImbalancer=True, useStratify=True)
else:
	X_train, y_train = getData(splitData=splitData, useImbalancer=False, useStratify=True)
	X_test, y_test = None, None

X_train, X_test, y_train, y_test = getAnnealingData()


def NeuralNetworkModel(splitData, X_train, X_test, y_train, y_test):
	clf = MLPClassifier(alpha=1e-4, max_iter=1000)
	layers = [(4, 6), (5, 7), (8, 10)]
	grid_values = {'hidden_layer_sizes': layers, 'activation': ['tanh', 'relu'], 'learning_rate': ['constant', 'invscaling']}
	grid_clf_acc = GridSearchCV(clf, param_grid=grid_values, scoring=['roc_auc', 'f1', 'accuracy'], refit='roc_auc')
	grid_clf_acc.fit(X_train, y_train.ravel())
	clf = grid_clf_acc.best_estimator_

	if splitData:
		y_preds = clf.predict(X_test)
		# printMetrics(y_test, y_preds)
		val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds)
	else:
		val_acc, val_pre, val_recall, val_auc, val_f1 = 0, 0, 0, 0, 0
	y_preds = clf.predict(X_train)
	acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds)

	val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
	metrics = (acc, pre, recall, auc, f1)
	# print("acc-" + str(acc) + "\tprecision-" + str(pre) + "\trecall-" + str(recall) + "\tauc-" + str(auc) + "\tf1-" + str(f1) + "\tval_accuracy-" + str(val_acc) + "\tval_precision-" + str(val_pre) + "\tval_recall-" + str(val_recall) + "\tval_auc-" + str(val_auc) + "\tval_f1-" + str(val_f1) + "\n")

	logAndSave(name_of_model="NeuralNetworkGS", clf=clf, metrics=metrics, val_metrics=val_metrics)


def NeuralNetworkModelV2(X_train, X_test, y_train, y_test):
	multi_class = True
	clf = MLPClassifier(alpha=1e-4, max_iter=1000)
	layers = [(4, 6), (5, 7), (8, 10), (256, 128, 64, 32, 16, 8)]
	grid_values = {'solver': ['adam', 'lbfgs'], 'hidden_layer_sizes': layers, 'activation': ['tanh', 'relu', 'logistic'], 'learning_rate': ['constant', 'invscaling']}
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

	logAndSaveV2(name_of_model="NeuralNetworkModelV2GS", clf=clf, metrics=metrics, val_metrics=val_metrics)


if __name__ == "__main__":
	# NeuralNetworkModel(splitData, X_train, X_test, y_train, y_test)
	NeuralNetworkModelV2(X_train, X_test, y_train, y_test)
