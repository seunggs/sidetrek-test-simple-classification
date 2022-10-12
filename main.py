# Importing all dependencies
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
from typing import Tuple

# Collecting data for classification model
def collect_data(n_samples: int = 1000,
                 n_features: int = 4,
                 n_informative: int = 2,
                 n_redundant: int = 0,
                 n_classes: int = 2,
                 random_state: int = 42) -> pd.DataFrame:
    ds = datasets.make_classification(n_samples = n_samples,
                                      n_features = n_features,
                                      n_informative = n_informative,
                                      n_redundant = n_redundant,
                                      n_classes = n_classes,
                                      random_state = random_state)
    return pd.DataFrame(ds[0], columns=["feature1",
                                        "feature2",
                                        "feature3",
                                        "feature4"]), np.ravel(pd.DataFrame(ds[1],
                                        columns=["target"]), order='C')

X, y = collect_data()

# Splitting X and y into train and test datasets
def split_train_test(feature: pd.DataFrame = X,
                     target: pd.DataFrame = y,
                     test_size: float = 0.33,
                     random_state: int = 42) -> Tuple[pd.DataFrame]:
    return train_test_split(feature,
                            target,
                            test_size = test_size,
                            random_state = random_state)

X_train, X_test, y_train, y_test = split_train_test()

# Creating the Multiple Linear Regression model
def build_model(model: RandomForestClassifier = RandomForestClassifier(),
                X_train: pd.DataFrame = X_train,
                y_train: pd.DataFrame = y_train) -> RandomForestClassifier:
    return model.fit(X_train, y_train)


# Predicting output using model
def predict_data(model: RandomForestClassifier = build_model(),
                 X_test: pd.DataFrame = X_test) -> np.ndarray:
    return model.predict(X_test)

y_pred = predict_data()

# Evaluating model on test data
def evaluate_model(y_test: pd.DataFrame = y_test,
                   y_pred: np.ndarray = y_pred) -> Tuple[float]:
    return metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), metrics.jaccard_score(y_test, y_pred)

acc, f1, prec, rec, js = evaluate_model()

print("Accuracy:",acc)
print("F1:",f1)
print("Precision:",prec)
print("Recall:",rec)
print("Jaccard Score:",js)