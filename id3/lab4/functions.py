from copy import copy
import numpy as np
import pandas as pd

from constants import *
from node import Node


def id3(X: pd.DataFrame, Y: pd.Series, max_depth: int) -> Node: # X - data for attributes, Y - class (model returns)
    if (Y.iloc[0] == Y).all():
        node = Node(value=int(Y.iloc[0]))
        return node

    if len(X.columns) == 0 or max_depth == 0:
        most_common_value = Y.mode().astype(int)
        node = Node(most_common_value)
        return node

    # calculate max inf gain
    best_attribute = calculate_max_inf_gain(X, Y)
    # print(best_attribute)
    node = Node(attr=best_attribute)
    for attribute_value in X[best_attribute].unique():
        # split data across values - f. ex. if attr has values 1, 2, 3 create children according to these values (1st child with 1, 2nd child with 2, 3rd child with 3)
        X_Y = copy(X)
        X_Y[Y.name] = Y
        X_Y_single_val = X_Y.loc[X_Y[best_attribute] == attribute_value]
        Y_after_split = X_Y_single_val[Y.name]
        X_after_split = X_Y_single_val.drop([Y.name, best_attribute], axis=1)
        node.add_child(attribute_value, id3(X_after_split, Y_after_split, max_depth - 1))
    return node


def predict_single(test_row: pd.Series, root: Node):
    if root.is_leaf():
        return root.value
    current_row_attr = root.attr
    current_val = test_row[current_row_attr]
    for attr_val in root.children:
        if current_val == attr_val:
            return predict_single(test_row, root.children.get(current_val))
    # if exact value not found
    attribute_values = root.children.keys()
    best_value = min(attribute_values, key=lambda x: abs(current_val - x))
    return predict_single(test_row, root.children.get(best_value))


def dataset_entropy(train_data: pd.Series) -> float: # train data - train dataset for cardio, label - label of dataset column, class set - values of dataset column
    total_val_amount = len(train_data)
    sum = 0
    for class_value in train_data.unique():
        class_count = train_data[train_data == class_value].count()
        class_entropy = class_count/total_val_amount*np.log2(class_count/total_val_amount)
        sum -= class_entropy
    return sum


def attribute_value_entropy(train_data_x_single_val: pd.DataFrame, train_data_y: pd.Series) -> float:
    # calculates entropy for single attribute value
    sum = 0
    dataset_values = train_data_y.unique() # cardio values
    for class_value in dataset_values:
        temp_sum = 0
        attr_val_count = len(train_data_x_single_val)
        try:
            class_value_count = train_data_x_single_val[train_data_y.name].value_counts()[class_value]
        except KeyError:
            class_value_count = 0
        if class_value_count != 0:
            temp_sum = class_value_count/attr_val_count * np.log2(class_value_count/attr_val_count)
        sum -= temp_sum
    return sum


def attribute_entropy(train_data_x: pd.DataFrame, train_data_y: pd.Series, attribute: str) -> float: #inf
    train_data = copy(train_data_x)
    train_data[train_data_y.name] = train_data_y
    attribute_values = train_data_x[attribute].unique()
    entropy = 0
    dataset_length = len(train_data_x)
    for val in attribute_values:
        train_data_x_single_val = train_data.loc[train_data[attribute] == val]
        value_probablity = len(train_data_x_single_val)/dataset_length
        temp_entropy = value_probablity * attribute_value_entropy(train_data_x_single_val, train_data_y)
        entropy += temp_entropy
    return entropy


def calculate_max_inf_gain(data_x: pd.DataFrame, data_y: pd.Series) -> str:
    entropy = dataset_entropy(data_y)
    best_attribute = None
    max_inf_gain = 0
    for attr in data_x.columns:
        inf_gain = entropy - attribute_entropy(data_x, data_y, attr)
        if inf_gain >= max_inf_gain:
            max_inf_gain = inf_gain
            best_attribute = attr
    return best_attribute


def fit_data(X, Y, max_depth):
    root = id3(X, Y, max_depth)
    return root


def predict_data(X: pd.DataFrame, root: Node):
    evaluated_set = X.apply(lambda row: predict_single(row, root), axis=1)
    return evaluated_set


def read_data(dataset: str, separator: str) -> pd.DataFrame:
    df = pd.read_csv(dataset, sep=separator)
    return df
