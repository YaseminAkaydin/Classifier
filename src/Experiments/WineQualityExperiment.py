import pandas as pd
import numpy as np

df = pd.read_csv('/Users/yaseminakaydin/PycharmProjects/pythonProject/Classifier/Data/winequality-red 3.csv')
print(df.dtypes)

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

print(df.head())

#Split the data
X = df.drop('quality', axis=1).copy()
y = df['quality'].copy()
y = le.fit_transform(y)
print(df['quality'].unique())

print(X[X.isna().any(axis=1)])
from src.Classifier.functions import DecisionClassifierScores, RandomForestScores, XGBoostScores
DecisionClassifierScores(X, y)
RandomForestScores(X, y)
XGBoostScores(X, y)



