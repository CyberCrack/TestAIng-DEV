import h2o
import numpy as np
from h2o.automl import H2OAutoML

from Utility import getAnnealingData, printMetrics, logAndSaveV2, getMetrics

X_train, X_test, y_train, y_test = getAnnealingData()
multi_class = True

h2o.init(ignore_config=True)
trainFrame = h2o.H2OFrame(np.concatenate((X_train, y_train.reshape(-1, 1)), axis=1))
testFrame = h2o.H2OFrame(np.concatenate((X_test, y_test.reshape(-1, 1)), axis=1))
x_labels = list(trainFrame.columns)
y_labels = x_labels[-1]
x_labels.remove(y_labels)
trainFrame[y_labels] = trainFrame[y_labels].asfactor()
testFrame[y_labels] = testFrame[y_labels].asfactor()

aml = H2OAutoML(max_runtime_secs=60)
aml.train(x=x_labels, y=y_labels, training_frame=trainFrame, validation_frame=testFrame)

y_predsFrame = aml.leader.predict(testFrame)
y_test_pred_df = y_predsFrame.as_data_frame()
y_predsFrame = aml.leader.predict(trainFrame)
y_train_pred_df = y_predsFrame.as_data_frame()

y_preds = y_test_pred_df['predict'].values
# printMetrics(y_true=y_test, y_pred=y_preds, multi_class=multi_class)
val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds, multi_class=multi_class)
# print("*" * 100)
y_preds = y_train_pred_df['predict'].values
# printMetrics(y_true=y_train, y_pred=y_preds, multi_class=multi_class)
acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds, multi_class=multi_class)
val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
metrics = (acc, pre, recall, auc, f1)
logAndSaveV2(name_of_model="H2OModelV2", clf=aml, metrics=metrics, val_metrics=val_metrics)
