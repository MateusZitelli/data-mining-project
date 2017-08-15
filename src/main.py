import autosklearn.classification
import autosklearn.metrics
import sklearn.datasets
import sklearn.metrics
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from time import gmtime, strftime

classes = {
    "CYT": 0,
    "NUC": 1,
    "MIT": 2,
    "ME3": 3,
    "ME2": 4,
    "ME1": 5,
    "EXC": 6,
    "VAC": 7,
    "POX": 8,
    "ERL": 9
}

attrs = ["name", "mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]
to_use = ["mcg", "gvh", "alm", "mit", "erl", "pox", "vac", "nuc", "class"]

v = pd.read_csv("./yeast.data", sep="\s+", usecols=to_use, names=attrs, converters={
    9: lambda c: classes[c]
})

X = v[to_use[:-1]]
y = v[to_use[-1]]

X_train, X_test, y_train, y_test = \
sklearn.model_selection.train_test_split(X, y, random_state=1)
automl = autosklearn.classification.AutoSklearnClassifier(
    delete_tmp_folder_after_terminate=False,
    delete_output_folder_after_terminate=False,
    time_left_for_this_task=60 * 120,
    per_run_time_limit=60 * 12,
    resampling_strategy='cv',
    resampling_strategy_arguments={'folds': 4},
    tmp_folder="./shared/tmp",
    output_folder="./shared/")

automl.fit(X_train.copy(), y_train.copy(), dataset_name='yeast')
automl.refit(np.asarray(X_train), np.asarray(y_train))
y_hat = automl.predict(X_test)
print(automl.show_models())
print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

joblib.dump(automl, './shared/automl-%s.pkl' % (strftime("%Y-%m-%d %H:%M:%S", gmtime()), ))
