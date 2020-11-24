import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os

path=os.path.join('data',"raw")

def prepare_data():
    fake = pd.read_csv(os.path.join(path, "Fake.csv")).drop('date', axis=1)
    fake["class"] = "Fake"

    real = pd.read_csv(os.path.join(path, "True.csv")).drop('date', axis=1)
    real["class"] = "Real"

    df = pd.concat([real, fake], sort=False).drop("subject", axis=1)
    return df

def train_test_prepare(df):
    X,y = df.drop(["class"],axis=1),df["class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    return X_train, X_test, y_train, y_test
