import time

from AdaBoost_Model import AdaBoostModel
from LogisticRegression_Model import LogisticRegressionModel
from NeuralNetwork_Model import NeuralNetworkModel
from RandomForest_Model import RandomForestModel
from SVM_Model import SVMModel
from Utility import getData
from XgBoost_Model import XGBClassifierModel


def TrainAllModels(splitData):
	for i in range(100):
		print("Current i: ", i)
		if splitData:
			X_train, X_test, y_train, y_test = getData(useImbalancer=True, useStratify=True)
		else:
			X_train, y_train = getData(splitData=splitData, useImbalancer=False, useStratify=True)
			X_test, y_test = None, None

		AdaBoostModel(splitData=splitData, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		LogisticRegressionModel(splitData=splitData, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		NeuralNetworkModel(splitData=splitData, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		RandomForestModel(splitData=splitData, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		SVMModel(splitData=splitData, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		XGBClassifierModel(splitData=splitData, X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


splitData = True
startTime = time.time()
TrainAllModels(splitData=splitData)
print("TimeTaken: ", time.time() - startTime)
splitData = False
startTime = time.time()
print("************************************************************************************************")
TrainAllModels(splitData=splitData)
print("TimeTaken: ", time.time() - startTime)
