from tpot import TPOTClassifier

from Utility import printMetrics, getAnnealingData

X_train, X_test, y_train, y_test = getAnnealingData()
multi_class = True

tpot = TPOTClassifier(max_time_mins=15, generations=100, population_size=100, scoring='f1_weighted', verbosity=3, random_state=101, cv=10, n_jobs=2)

tpot.fit(X_train, y_train)

# print(tpot.score(X_test, y_test))
y_pred = tpot.predict(X_test)

# printMetrics(y_test, y_pred, multi_class=multi_class)
y_pred = tpot.predict(X_train)
# print("*" * 100)
# printMetrics(y_train, y_pred, multi_class=multi_class)
tpot.export('tpot_classifier_pipeline_v2.py')
