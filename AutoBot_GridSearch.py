import time

from AdaBoost_Model_GridSearch import AdaBoostModel
from LogisticRegression_Model_GridSearch import LogisticRegressionModel
from NeuralNetwork_Model_GridSearch import NeuralNetworkModel
from RandomForest_Model_GridSearch import RandomForestModel
from SVM_Model_GridSearch import SVMModel
from Utility import getData
from XgBoost_Model_GridSearch import XGBClassifierModel


def TrainAllModels(splitData):
	for i in range(5):
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
