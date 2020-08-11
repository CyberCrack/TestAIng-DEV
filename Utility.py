import datetime
import os
import pickle

import h2o.automl
import numpy as np
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SVMSMOTE, SMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from sklearn import preprocessing, compose
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

standard_scaler = None
column_transformer_pipeline = None
Model_num = ''


def getAnnealingData():
	global Model_num, column_transformer_pipeline

	def replaceUnknows(data):
		data['family'] = data['family'].replace(to_replace='?', value='UNK')
		data['product-type'] = data['product-type'].replace(to_replace='C', value=1).apply(pd.to_numeric)
		data['steel'] = data['steel'].replace(to_replace='?', value='NA')
		data['temper_rolling'] = data['temper_rolling'].replace(to_replace='?', value='NA')
		data['condition'] = data['condition'].replace(to_replace='?', value='NA')
		data['formability'] = data['formability'].replace(to_replace='?', value='0')
		data['non-ageing'] = data['non-ageing'].replace(to_replace='?', value='NA')
		data['surface-finish'] = data['surface-finish'].replace(to_replace='?', value='NA')
		data['surface-quality'] = data['surface-quality'].replace(to_replace='?', value='NA')
		data['enamelability'] = data['enamelability'].replace(to_replace='?', value='0')
		data['bc'] = data['bc'].replace(to_replace='?', value='NA')
		data['bf'] = data['bf'].replace(to_replace='?', value='NA')
		data['bt'] = data['bt'].replace(to_replace='?', value='NA')
		data['bw/me'] = data['bw/me'].replace(to_replace='?', value='NA')
		data['bl'] = data['bl'].replace(to_replace='?', value='NA')
		data['m'] = data['m'].replace(to_replace='?', value=0).apply(pd.to_numeric)
		data['chrom'] = data['chrom'].replace(to_replace='?', value='NA')
		data['phos'] = data['phos'].replace(to_replace='?', value='NA')
		data['cbond'] = data['cbond'].replace(to_replace='?', value='NA')
		data['marvi'] = data['marvi'].replace(to_replace='?', value=0).apply(pd.to_numeric)
		data['exptl'] = data['exptl'].replace(to_replace='?', value='NA')
		data['ferro'] = data['ferro'].replace(to_replace='?', value='NA')
		data['corr'] = data['corr'].replace(to_replace='?', value=0).apply(pd.to_numeric)
		data['exptl'] = data['exptl'].replace(to_replace='?', value='NA')
		data['blue/bright/varn/clean'] = data['blue/bright/varn/clean'].replace(to_replace='?', value='NA')
		data['lustre'] = data['lustre'].replace(to_replace='?', value='NA')
		data['jurofm'] = data['jurofm'].replace(to_replace='?', value=0).apply(pd.to_numeric)
		data['s'] = data['s'].replace(to_replace='?', value=0).apply(pd.to_numeric)
		data['p'] = data['p'].replace(to_replace='?', value=0).apply(pd.to_numeric)
		data['oil'] = data['oil'].replace(to_replace='?', value='NA')
		data['packing'] = data['packing'].replace(to_replace='?', value=0).apply(pd.to_numeric)
		return data

	dataSource = 'DataSource/annealing.csv'
	testDataSource = 'DataSource/annealing-TEST.csv'
	data = pd.read_csv(dataSource, header=None)
	testData = pd.read_csv(testDataSource, header=None)

	col_headings = ['family', 'product-type', 'steel', 'carbon', 'hardness', 'temper_rolling', 'condition', 'formability', 'strength', 'non-ageing', 'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt', 'bw/me', 'bl', 'm', 'chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue/bright/varn/clean', 'lustre', 'jurofm', 's', 'p', 'shape', 'thick', 'width', 'len', 'oil', 'bore', 'packing', 'target']
	col_index = {0: 'family', 1: 'product-type', 2: 'steel', 3: 'carbon', 4: 'hardness', 5: 'temper_rolling', 6: 'condition', 7: 'formability', 8: 'strength', 9: 'non-ageing', 10: 'surface-finish', 11: 'surface-quality', 12: 'enamelability', 13: 'bc', 14: 'bf', 15: 'bt', 16: 'bw/me', 17: 'bl', 18: 'm', 19: 'chrom', 20: 'phos', 21: 'cbond', 22: 'marvi', 23: 'exptl', 24: 'ferro', 25: 'corr', 26: 'blue/bright/varn/clean', 27: 'lustre', 28: 'jurofm', 29: 's', 30: 'p', 31: 'shape', 32: 'thick', 33: 'width', 34: 'len', 35: 'oil', 36: 'bore', 37: 'packing', 38: 'target'}
	col_to_drop = ['family', 'product-type', 'non-ageing', 'surface-finish', 'enamelability', 'bc', 'm', 'chrom', 'phos', 'cbond', 'marvi', 'exptl', 'ferro', 'corr', 'blue/bright/varn/clean', 'lustre', 'jurofm', 's', 'p']
	# col_to_drop = []
	data.columns = col_headings
	testData.columns = col_headings

	data = replaceUnknows(data)
	testData = replaceUnknows(testData)

	data = data.drop(col_to_drop, axis=1)
	testData = testData.drop(col_to_drop, axis=1)
	X_train = data.drop('target', axis=1)
	y_train = data['target'].values
	lable_enc = LabelEncoder()
	y_train = lable_enc.fit_transform(y_train)
	cols_to_oneHotEncode = ['family', 'steel', 'temper_rolling', 'condition', 'formability', 'non-ageing', 'surface-finish', 'surface-quality', 'enamelability', 'bc', 'bf', 'bt', 'bw/me', 'bl', 'chrom', 'phos', 'cbond', 'exptl', 'ferro', 'blue/bright/varn/clean', 'lustre', 'shape', 'oil', 'packing']
	cols_to_oneHotEncode = list(set(list(X_train.columns)).intersection(set(cols_to_oneHotEncode)))
	cols_to_scale = ['product-type', 'carbon', 'hardness', 'strength', 'thick', 'width', 'len', 'bore', 'm', 'jurofm', 'p', 'marvi', 's', 'corr']
	cols_to_scale = list(set(list(X_train.columns)).intersection(set(cols_to_scale)))
	column_transformer_pipeline = make_column_transformer((OneHotEncoder(drop='first'), cols_to_oneHotEncode),
														  (StandardScaler(), cols_to_scale),
														  remainder='passthrough')

	X_train = column_transformer_pipeline.fit_transform(X=X_train)
	X_test = testData.drop('target', axis=1)
	X_test = column_transformer_pipeline.transform(X=X_test)
	y_test = testData['target'].values
	y_test = lable_enc.transform(y=y_test)
	Model_num = str(abs(hash(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))))

	return X_train, X_test, y_train, y_test


def logAndSaveV2(name_of_model, clf, metrics, val_metrics):
	global Model_num, column_transformer_pipeline
	acc, pre, recall, auc, f1 = metrics
	val_acc, val_pre, val_recall, val_auc, val_f1 = val_metrics
	name_of_model = name_of_model

	msg = Model_num + "-" + name_of_model + "\t\t" + "acc-" + str(acc) + "\tprecision-" + str(pre) + "\trecall-" + str(recall) + "\tauc-" + str(auc) + "\tf1-" + str(f1) + "\tval_accuracy-" + str(val_acc) + "\tval_precision-" + str(val_pre) + "\tval_recall-" + str(val_recall) + "\tval_auc-" + str(val_auc) + "\tval_f1-" + str(val_f1) + "\n"
	f = open("SKlogsV2.log", "a+")
	f.write(msg)
	f.close()
	if not os.path.exists("SKMetrics.csv"):
		f = open("SKMetrics.csv", "w")
		f.write(",".join(["Model No.", "Model Type", "Accuracy", "Precision", "Recall", "AUC", "F1", "Val_Accuracy", "Val_Precision", "Val_Recall", "Val_AUC", "Val_F1"]) + "\n")
		f.close()

	f = open("SKMetrics.csv", "a+")
	msg = ",".join([Model_num, name_of_model, str(acc), str(pre), str(recall), str(auc), str(f1), str(val_acc), str(val_pre), str(val_recall), str(val_auc), str(val_f1)])
	f.write(msg + "\n")
	f.close()
	if not os.path.exists("ColumnTransformerV2"):
		os.mkdir("ColumnTransformerV2")
	name_of_file = "_".join([Model_num, name_of_model, "ColumnTransformer"]) + ".pickle"
	if type(column_transformer_pipeline) is compose._column_transformer.ColumnTransformer:
		pickle_out = open(os.path.join("ColumnTransformerV2", name_of_file), "wb")
		pickle.dump(column_transformer_pipeline, pickle_out)
	name_of_file = "_".join([Model_num, name_of_model]) + ".pickle"
	if not os.path.exists("SKLearnModels"):
		os.mkdir("SKLearnModels")
	if not os.path.exists("H2OModels"):
		os.mkdir("H2OModels")
	if not os.path.exists("AutoKerasModels"):
		os.mkdir("AutoKerasModels")
	if type(clf) is xgb.XGBClassifier:
		name_of_file = "_".join([Model_num, name_of_model]) + ".bin"
		clf.save_model(os.path.join("SKLearnModels", name_of_file))
	elif isinstance(clf, h2o.automl.H2OAutoML):
		name_of_file = "_".join([Model_num, name_of_model])
		h2o.save_model(clf.leader, path=os.path.join("H2OModels", name_of_file))
	elif "autokeras" in str(type(clf)):
		name_of_file = "_".join([Model_num, name_of_model])
		model = clf.export_model()
		try:
			model.save(os.path.join("AutoKerasModels", name_of_file), save_format="tf")
		except:
			model.save(os.path.join("AutoKerasModels", name_of_file) + ".h5")
	else:
		pickle_out = open(os.path.join("SKLearnModels", name_of_file), "wb")
		pickle.dump(clf, pickle_out)


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


def printMetrics(y_true, y_pred, multi_class=False):
	con_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
	print(con_mat)
	accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
	print("accuracy: ", accuracy)
	if multi_class:
		precision = precision_score(y_true, y_pred, zero_division=0, average='weighted')
		recall = recall_score(y_true, y_pred, zero_division=0, average='weighted')
		ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')

		y_true_new = ohe.fit_transform(y_true.reshape(-1, 1))
		y_pred_new = ohe.transform(y_pred.reshape(-1, 1))
		roc_auc = roc_auc_score(y_true_new, y_pred_new, average='weighted', multi_class='ovr')
		f1score = f1_score(y_true=y_true, y_pred=y_pred, average='weighted')
	else:
		precision = precision_score(y_true, y_pred, zero_division=0)
		recall = recall_score(y_true, y_pred)
		roc_auc = roc_auc_score(y_true, y_pred)
		f1score = f1_score(y_true=y_true, y_pred=y_pred)
	print("precision: ", precision)

	print("recall: ", recall)

	print("roc_auc: ", roc_auc)

	print("f1score: ", f1score)


def getMetrics(y_true, y_pred, multi_class=False):
	if multi_class:
		ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
		y_true_new = ohe.fit_transform(y_true.reshape(-1, 1))
		y_test_new = ohe.transform(y_pred.reshape(-1, 1))

		return round(accuracy_score(y_true, y_pred), 4), round(precision_score(y_true, y_pred, zero_division=0, average='weighted'), 4), round(recall_score(y_true, y_pred, average='weighted'), 4), round(roc_auc_score(y_true_new, y_test_new, average='weighted', multi_class='ovr'), 4), round(f1_score(y_true=y_true, y_pred=y_pred, average='weighted'), 4)

	else:
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
