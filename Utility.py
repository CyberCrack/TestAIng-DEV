import datetime
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SVMSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split

standard_scaler = None


def getData(splitData=True, useImbalancer=False, useStratify=False):
	global standard_scaler
	data = pd.read_csv(filepath_or_buffer="DataSource/binary.csv")
	X = data.values[:, 1:-1]
	rank_dummy = pd.get_dummies(data['rank'], drop_first=True).to_numpy()
	X = np.concatenate((X, rank_dummy), axis=1)
	y = data.values[:, 0].reshape(-1, 1)
	if useStratify:
		stratify = y
	else:
		stratify = None
	if splitData:
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101, shuffle=True, stratify=stratify)
	else:
		X_train = X
		y_train = y
	if useImbalancer and splitData:
		tl = TomekLinks(sampling_strategy='majority')
		X_train, y_train = tl.fit_sample(X=X_train, y=y_train)
		# print("After 1st pass: ", len(X_train))
		svmsm = SVMSMOTE(sampling_strategy='minority', k_neighbors=10)
		X_train, y_train = svmsm.fit_sample(X=X_train, y=y_train)
		# print("After 2nd pass: ", len(X_train))
		rus = RandomUnderSampler(sampling_strategy='all', )
		X_train, y_train = rus.fit_sample(X=X_train, y=y_train)
		# print("After 3rd pass: ", len(X_train))
		tl = TomekLinks(sampling_strategy='all')
		X_train, y_train = tl.fit_sample(X=X_train, y=y_train)
		# print("After 4th pass: ", len(X_train))
		smote = SMOTE(sampling_strategy='minority', k_neighbors=10)
		X_train, y_train = smote.fit_sample(X=X_train, y=y_train)
		# print("After 5th pass: ", len(X_train))
		tl = TomekLinks(sampling_strategy='all')
		X_train, y_train = tl.fit_sample(X=X_train, y=y_train)
	# print("After 6th pass: ", len(X_train))

	standard_scaler = preprocessing.StandardScaler()
	X_train = standard_scaler.fit_transform(X_train)
	if splitData:
		X_test = standard_scaler.transform(X_test)
	# print(X_train[:5,:])
	# print(X_test[:5,:])
	unique, counts = np.unique(y_train, return_counts=True)
	# print(unique, counts)
	# print("y_train\n", np.asarray((unique, counts)).T)
	if splitData:
		unique, counts = np.unique(y_test, return_counts=True)
	# print("y_test\n", np.asarray((unique, counts)).T)
	if splitData:
		return X_train, X_test, y_train.ravel(), y_test.ravel()
	else:
		return X_train, y_train.ravel()


def printMetrics(y_true, y_pred):
	con_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
	print(con_mat)
	accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
	print("accuracy: ", accuracy)
	precision = precision_score(y_true, y_pred, zero_division=0)
	print("precision: ", precision)
	recall = recall_score(y_true, y_pred)
	print("recall: ", recall)
	roc_auc = roc_auc_score(y_true, y_pred)
	print("roc_auc: ", roc_auc)
	f1score = f1_score(y_true=y_true, y_pred=y_pred)
	print("f1score: ", f1score)


def getMetrics(y_true, y_pred):
	return round(accuracy_score(y_true, y_pred), 4), round(precision_score(y_true, y_pred, zero_division=0), 4), round(recall_score(y_true, y_pred), 4), round(roc_auc_score(y_true, y_pred), 4), round(f1_score(y_true, y_pred, zero_division=0), 4)


def logAndSave(name_of_model, clf, metrics, val_metrics):
	global standard_scaler
	acc, pre, recall, auc, f1 = metrics
	val_acc, val_pre, val_recall, val_auc, val_f1 = val_metrics
	name_of_model = name_of_model
	datetime_of_creation = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
	msg = str(abs(hash(datetime_of_creation))) + "-" + name_of_model + "\t\t" + "acc-" + str(acc) + "\tprecision-" + str(pre) + "\trecall-" + str(recall) + "\tauc-" + str(auc) + "\tf1-" + str(f1) + "\tval_accuracy-" + str(val_acc) + "\tval_precision-" + str(val_pre) + "\tval_recall-" + str(val_recall) + "\tval_auc-" + str(val_auc) + "\tval_f1-" + str(val_f1) + "\n"
	f = open("SKlogs.log", "a+")
	f.write(msg)
	f.close()
	if not os.path.exists("SKMetrics.csv"):
		f = open("SKMetrics.csv", "w")
		f.write(",".join(["Model No.", "Model Type", "Accuracy", "Precision", "Recall", "AUC", "F1", "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_AUC", "Val_F1"]) + "\n")
		f.close()

	f = open("SKMetrics.csv", "a+")
	msg = ",".join([str(abs(hash(datetime_of_creation))), name_of_model, str(acc), str(pre), str(recall), str(auc), str(f1), str(val_acc), str(val_pre), str(val_recall), str(val_auc), str(val_f1)])
	f.write(msg + "\n")
	f.close()
	if not os.path.exists("Scaler"):
		os.mkdir("Scaler")
	name_of_file = "_".join([str(abs(hash(datetime_of_creation))), name_of_model, "Scaler", datetime_of_creation]) + ".pickle"
	if type(standard_scaler) is preprocessing.StandardScaler:
		pickle_out = open(os.path.join("Scaler", name_of_file), "wb")
		pickle.dump(standard_scaler, pickle_out)
	name_of_file = "_".join([str(abs(hash(datetime_of_creation))), name_of_model, datetime_of_creation]) + ".pickle"
	if not os.path.exists("SKLearnModels"):
		os.mkdir("SKLearnModels")

	if type(clf) is xgb.XGBClassifier:
		name_of_file = "_".join([str(abs(hash(datetime_of_creation))), name_of_model, datetime_of_creation]) + ".bin"
		clf.save_model(os.path.join("SKLearnModels", name_of_file))
	else:
		pickle_out = open(os.path.join("SKLearnModels", name_of_file), "wb")
		pickle.dump(clf, pickle_out)


if __name__ == "__main__":
	getData(useImbalancer=True, useStratify=True)
