import autokeras as ak
import tensorflow as tf

from Utility import getAnnealingData, printMetrics, getMetrics, logAndSaveV2

X_train, X_test, y_train, y_test = getAnnealingData()
multi_class = True

clf = ak.StructuredDataClassifier(multi_label=True, metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()], overwrite=True, max_trials=20)
clf.fit(x=X_train, y=y_train, epochs=15, validation_data=(X_test, y_test))

model = clf.export_model()

y_preds = clf.predict(X_test).ravel()
printMetrics(y_true=y_test, y_pred=y_preds, multi_class=multi_class)
val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_true=y_test, y_pred=y_preds, multi_class=multi_class)

y_preds = clf.predict(X_train).ravel()
printMetrics(y_true=y_train, y_pred=y_preds, multi_class=multi_class)
acc, pre, recall, auc, f1 = getMetrics(y_true=y_train, y_pred=y_preds, multi_class=multi_class)

val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
metrics = (acc, pre, recall, auc, f1)

logAndSaveV2(name_of_model="AutoKerasV2", clf=clf, metrics=metrics, val_metrics=val_metrics)
