# Importing all dependencies
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import sklearn.metrics as metrics
import pandas as pd
import numpy as np

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


# Splitting X and y into train and test datasets
def split_train_test(feature: pd.DataFrame = collect_data()[0],
                     target: pd.DataFrame = collect_data()[1],
                     test_size: float = 0.33,
                     random_state: int = 42) -> pd.DataFrame:
    return train_test_split(feature,
                            target,
                            test_size = test_size,
                            random_state = random_state)

# Creating the Multiple Linear Regression model
def build_model(model: RandomForestClassifier = RandomForestClassifier(),
                X_train: pd.DataFrame = split_train_test()[0],
                y_train: pd.DataFrame = split_train_test()[2]) -> RandomForestClassifier:
    return model.fit(X_train, y_train)


# Predicting output using model
def predict_data(model: RandomForestClassifier = build_model(),
                 X_test: pd.DataFrame = split_train_test()[1]) -> np.ndarray:
    return model.predict(X_test)


# Evaluating model on test data
def evaluate_model(y_test: pd.DataFrame = split_train_test()[3],
                   y_pred: np.ndarray = predict_data()) -> tuple:
    return metrics.accuracy_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), metrics.jaccard_score(y_test, y_pred)

print("Accuracy:",evaluate_model()[0])
print("F1:",evaluate_model()[1])
print("Precision:",evaluate_model()[2])
print("Recall:",evaluate_model()[3])
print("Jaccard Score",evaluate_model()[4])