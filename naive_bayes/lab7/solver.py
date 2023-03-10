from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split, KFold
import numpy as np
import pandas as pd

from dataset_functions import *


class Solver(ABC):
    """A solver."""

    @abstractmethod
    def fit(self, X, y):
        """
        A method that fits the solver to the given data.
        X is the dataset without the class attribute.
        y contains the class attribute for each sample from X.
        It may return anything.
        """
        ...

    def predict(self, X):
        """
        A method that returns predicted class for each row of X
        """


class NaiveBayes(Solver):
    def __init__(self, laplace_l: float=1) -> None:
        self.laplace_l = laplace_l
        self.prior_prob = {}
        self.posteriori_prob = {}

    def fit(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self.y_column_name = y.name
        self.y_column = y
        total_val_amount = len(self.y_column)
        self.dataset_values = self.y_column.unique()
        training_data = X
        self.attributes = X.columns

        for class_val in self.dataset_values:
            class_count = self.y_column[self.y_column == class_val].count()
            self.prior_prob[class_val] = class_count/total_val_amount

        for class_val in self.dataset_values:
            class_count = self.y_column[self.y_column == class_val].count()
            class_val_dict = {}
            for attr in training_data.columns:
                attr_val_dict = {}
                for attr_val in training_data[attr].unique():
                    # class count && atttribute count + l / total class count + l * unique attr value count
                    attr_val_count = training_data.loc[(training_data[attr] == attr_val) & (self.y_column == class_val)].shape[0] + self.laplace_l
                    attr_val_prob = attr_val_count / (class_count + self.laplace_l * training_data[attr].nunique())
                    temp_dict = {attr_val: attr_val_prob}
                    attr_val_dict.update(temp_dict)
                temp_attr_dict = {attr: attr_val_dict}
                class_val_dict.update(temp_attr_dict)
            self.posteriori_prob[class_val] = class_val_dict
        return self.posteriori_prob

    def predict_single(self, row: pd.Series):
        prob_scores = {}
        prob_sum = 0

        for class_val in self.dataset_values:
            prop_val = self.prior_prob[class_val]
            for attr in self.attributes:
                data_val = row[attr]
                try:
                    prop_val *= self.posteriori_prob[class_val][attr][data_val]
                except KeyError:
                    continue
            prob_sum += prop_val
            prob_scores[class_val] = prop_val

        for class_val in self.dataset_values:
            prob_scores[class_val] /= prob_sum

        prediction = max(prob_scores, key=prob_scores.get)
        return prediction


    def predict(self, X: pd.DataFrame):
        evaluated_set = X.apply(lambda row: self.predict_single(row), axis=1)
        return evaluated_set


def k_cross_validation(X: pd.DataFrame, y: pd.Series, k: int=5) -> list:
    kf = KFold(n_splits=k)
    naive_bayes = NaiveBayes()
    results = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        naive_bayes.fit(X_train, y_train)
        print("Finished training.")
        print("Validate: ")
        prediction = naive_bayes.predict(X_test)
        result = prediction.eq(y_test)
        accuracy = result.sum() / len(result)
        results.append(accuracy)
        print(f"Result: { accuracy}")
    return results


def train_test_split_validation(data: pd.DataFrame, n_iter: int) -> list:
    results = []
    for i in range(n_iter):
        x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(data)
        print("Start training.")
        naive_bayes = NaiveBayes()
        naive_bayes.fit(x_train, y_train)
        print("Validate: ")
        prediction = naive_bayes.predict(x_valid)
        result = prediction.compare(y_valid, keep_shape=True)
        accuracy = result['self'].isna().sum() / len(result)
        results.append(accuracy)
        print(f"Result:  {accuracy}")
    return results



def main():
    df = prepare_dataset("cardio_train.csv")
    y_column = df["cardio"]
    # columns_to_remove_outliers = ["age", "weight", "height", "ap_hi", "ap_lo"]
    # df = remove_outliers(df, columns_to_remove_outliers)
    split_results = train_test_split_validation(df, 5)
    df = df.drop("cardio", axis=1)
    cross_results = k_cross_validation(df, y_column)
    print(f"Split results: {split_results}")
    print(f"K-fold results: {cross_results}")


if __name__ == "__main__":
    main()