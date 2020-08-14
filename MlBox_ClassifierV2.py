from mlbox.model import classification
from mlbox.optimisation import *

from Utility import getAnnealingData, printMetrics, getMetrics, logAndSaveV2

import pandas as pd

X_train, X_test, y_train, y_test = getAnnealingData()
multi_class = True
opt = Optimiser(scoring="f1_weighted")
space = {
	'fs__strategy': {"search": "choice", "space": ["variance", "rf_feature_importance", "l1"]},
	'est__strategy': {"search": "choice", "space": ["LightGBM", "RandomForest", "ExtraTrees", "Tree", "Bagging", "AdaBoost", "Linear"]}
}
df = {"train": pd.DataFrame(X_train), "target": pd.Series(y_train)}
best = opt.optimise(space, df, 21)


clf_feature_selector = classification.Clf_feature_selector(strategy=best['fs__strategy'])
newDf = clf_feature_selector.fit_transform(df['train'], df['target'])
testdf = clf_feature_selector.transform(pd.DataFrame(X_test))

clf = classification.Classifier(strategy=best['est__strategy'])
clf.fit(newDf, df['target'])
y_preds = clf.predict(testdf)
# printMetrics(y_true=y_test, y_pred=y_preds, multi_class=multi_class)
val_acc, val_pre, val_recall, val_auc, val_f1 = getMetrics(y_test, y_preds, multi_class=multi_class)
y_preds = clf.predict(newDf)
# printMetrics(y_true=y_train, y_pred=y_preds, multi_class=multi_class)
acc, pre, recall, auc, f1 = getMetrics(y_train, y_preds, multi_class=multi_class)
val_metrics = (val_acc, val_pre, val_recall, val_auc, val_f1)
metrics = (acc, pre, recall, auc, f1)
logAndSaveV2(name_of_model="MlBoxV2", clf=clf, metrics=metrics, val_metrics=val_metrics)
