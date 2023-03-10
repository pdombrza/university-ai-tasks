from copy import copy
import pandas as pd
from sklearn.model_selection import train_test_split

from constants import *
from functions import read_data


def discretize_data(df: pd.DataFrame) -> None:
    df['age'] = df['age'].apply(lambda x: x // AGE_DIVIDE) # discretize age - when divided by 3000 5 distinct values remain
    df['height'] = df['height'].apply(lambda x: x // HEIGTH_DIVIDE) # discretize height - when divided by 50 5 distint values remain
    df['weight'] = df['weight'].apply(lambda x: int(x // WEIGHT_DIVIDE)) # discretize weight - when divided by 40, 6 values remain
    df['ap_hi'] = df['ap_hi'].apply(discretize_ap_hi)
    df['ap_lo'] = df['ap_lo'].apply(discretize_ap_lo)


def split_data(df: pd.DataFrame):
    x = copy(df.drop(columns=['cardio']))
    y = df['cardio']
    x_train, x_rem, y_train, y_rem = train_test_split(x, y, train_size=TRAIN_SIZE)
    x_valid, x_test, y_valid, y_test = train_test_split(x_rem, y_rem, test_size=TEST_SIZE)
    return x_train, y_train, x_valid, y_valid, x_test, y_test


def discretize_ap_hi(x):
    if x < 120:
        return 0
    elif x < 130:
        return 1
    elif x < 140:
        return 2
    elif x < 180:
        return 3
    return 4


def discretize_ap_lo(x):
    if x < 80:
        return 0
    elif x < 90:
        return 1
    elif x < 120:
        return 2
    return 3


def prepare_dataset(dataset: str, separator: str=';') -> pd.DataFrame:
    df = read_data(dataset, separator)
    df = df.drop('id', axis=1)
    discretize_data(df)
    return df