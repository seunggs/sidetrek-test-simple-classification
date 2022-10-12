# Importing all dependencies
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import pandas as pd
import numpy as np
import typing
from dataclasses import dataclass


@dataclass
class Hyperparameters(object):
    """
    n_samples
    n_features
    n_informative
    n_redundant
    n_classes
    random_state
    feature
    target
    test_size
    model
    """

    n_samples: int = 1000
    n_features: int = 4
    n_informative: int = 2
    n_redundant: int = 0
    n_classes: int = 2
    random_state: int = 42
    test_size: float = 0.33
    model: RandomForestClassifier = RandomForestClassifier()
    

# Collecting data for classification model
def collect_data(n_samples: int,
                 n_features: int,
                 n_informative: int,
                 n_redundant: int,
                 n_classes: int,
                 random_state: int) -> pd.DataFrame:
    """
    n_samples: int = 1000,
    n_features: int = 4,
    n_informative: int = 2,
    n_redundant: int = 0,
    n_classes: int = 2,
    random_state: int = 42
    """
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
def split_train_test(feature: pd.DataFrame,
                     target: pd.DataFrame,
                     test_size: float,
                     random_state: int) -> typing.Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    feature: pd.DataFrame = X,
    target: pd.DataFrame = y,
    test_size: float = 0.33,
    random_state: int = 42
    """
    return train_test_split(feature,
                            target,
                            test_size = test_size,
                            random_state = random_state)

X_train, X_test, y_train, y_test = split_train_test()

# Creating the Multiple Linear Regression model
def build_model(model: RandomForestClassifier,
                X_train: pd.DataFrame,
                y_train: pd.DataFrame) -> RandomForestClassifier:
    """
    model: RandomForestClassifier = RandomForestClassifier(),
    X_train: pd.DataFrame = X_train,
    y_train: pd.DataFrame = y_train
    """
    return model.fit(X_train, y_train)

clf = build_model()

# Predicting output using model
def predict_data(model: RandomForestClassifier,
                 X_test: pd.DataFrame) -> np.ndarray:
    """
    model: RandomForestClassifier = clf,
    X_test: pd.DataFrame = X_test)
    """
    return model.predict(X_test)

y_pred = predict_data()

# Evaluating model on test data
def evaluate_model(y_test: pd.DataFrame,
                   y_pred: np.ndarray) -> typing.Tuple[float, float, float, float, float]:
    """
    y_test: pd.DataFrame = y_test,
    y_pred: np.ndarray = y_pred
    """
    return metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), metrics.jaccard_score(y_test, y_pred)

acc, f1, prec, rec, js = evaluate_model()

print("Accuracy:",acc)
print("F1:",f1)
print("Precision:",prec)
print("Recall:",rec)
print("Jaccard Score:",js)