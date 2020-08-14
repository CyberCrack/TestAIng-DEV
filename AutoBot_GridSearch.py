import time

from AdaBoost_Model_GridSearch import AdaBoostModel, AdaBoostModelV2
from LogisticRegression_Model_GridSearch import LogisticRegressionModel, LogisticRegressionModelV2
from NeuralNetwork_Model_GridSearch import NeuralNetworkModel, NeuralNetworkModelV2
from RandomForest_Model_GridSearch import RandomForestModel, RandomForestModelV2
from SVM_Model_GridSearch import SVMModel, SVMModelV2
from Utility import getData, getAnnealingData
from XgBoost_Model_GridSearch import XGBClassifierModel, XGBClassifierModelV2


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


def TrainAllModelsV2():
	for i in range(5):
		print("Current i: ", i)
		X_train, X_test, y_train, y_test = getAnnealingData()

		AdaBoostModelV2(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		LogisticRegressionModelV2(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		NeuralNetworkModelV2(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		RandomForestModelV2(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		SVMModelV2(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)
		XGBClassifierModelV2(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)


splitData = True
startTime = time.time()
TrainAllModels(splitData=splitData)
print("TimeTaken: ", time.time() - startTime)
splitData = False
startTime = time.time()
print("************************************************************************************************")
TrainAllModels(splitData=splitData)
print("TimeTaken: ", time.time() - startTime)

# For V2
startTime = time.time()
TrainAllModelsV2()
print("TimeTaken: ", time.time() - startTime)