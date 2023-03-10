from abc import ABC, abstractmethod

import pandas as pd
from functions import *
from dataset_functions import *


class Solver(ABC):
    """A solver. Parameters may be passed during initialization."""
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters

    @abstractmethod
    def get_parameters(self):
        """Returns a dictionary of hyperparameters"""
        ...

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


class ID3(Solver):
    def __init__(self, parameters: dict) -> None:
        super().__init__(parameters)
        self.root = None

    def get_parameters(self):
        return self.parameters

    def fit(self, X, y):
        root = id3(X, y, self.parameters["max depth"])
        self.root = root

    def predict(self, X):
        evaluated_set = X.apply(lambda row: predict_single(row, self.root), axis=1)
        return evaluated_set


def build_and_validate(data_after_split):
    x_train = data_after_split["x_train"]
    y_train = data_after_split["y_train"]
    x_valid = data_after_split["x_valid"]
    y_valid = data_after_split["y_valid"]
    for max_depth in range(0, 12):
        parameters = {
            "max depth": max_depth
        }
        print("Start training.")
        solver = ID3(parameters)
        solver.fit(x_train, y_train)
        print("Tree built. Validate: ")
        prediction = solver.predict(x_valid)
        prediction_cleaned = prediction.dropna(axis=1, how='any').squeeze()
        result = prediction_cleaned.compare(y_valid, keep_shape=True)
        print(f"Depth: {max_depth}")
        print(f"Result:  { result['self'].isna().sum() / len(result)}")


def test_model(data_after_split: dict, solver: ID3) -> None:
    x_train = data_after_split["x_train"]
    y_train = data_after_split["y_train"]
    x_test = data_after_split["x_test"]
    y_test = data_after_split["y_test"]
    solver.fit(x_train, y_train)
    max_depth = solver.get_parameters()["max depth"]
    prediction = solver.predict(x_test)
    prediction_cleaned = prediction.dropna(axis=1, how='any').squeeze()
    result = prediction_cleaned.compare(y_test, keep_shape=True)
    print(f"Depth : {max_depth}\nResult:  { result['self'].isna().sum() / len(result)}")



def main():
    dataset = "cardio_train.csv"
    data = prepare_dataset(dataset)
    x_train, y_train, x_valid, y_valid, x_test, y_test = split_data(data)
    after_split = {
        "x_train": x_train,
        "y_train": y_train,
        "x_valid": x_valid,
        "y_valid": y_valid,
        "x_test": x_test,
        "y_test": y_test
    }
    parameters = {
        "max depth": 4
    }
    solver = ID3(parameters)
    test_model(after_split, solver)


if __name__ == "__main__":
    main()