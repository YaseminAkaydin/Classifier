import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from src.Classifier.functions import DecisionClassifierScores, RandomForestScores, XGBoostScores, \
    RandomizedSearch, createLernCurve, createValidCurve, f1ScoreValidCurve

df=pd.read_csv("/Users/yaseminakaydin/PycharmProjects/pythonProject/Classifier/Data/adult 2.csv", header=0)
df.isnull().sum()
print(df.info)
# Replace All Null Data in NaN
df = df.fillna(np.nan)
df['income'].replace(['<=50K','>50K'],[0, 1],inplace=True)
print(df.dtypes)
object_cols = df.select_dtypes(include='object').columns
df= pd.get_dummies(df, columns=object_cols)

X = df.drop('income', axis=1).copy()
y = df['income'].copy()


X.replace('?', np.nan, inplace=True)
X.dropna(inplace=True)
missing_values = X.isna().sum()




clf_dt_model, X_train_dt, y_train_dt, X_test_dt, y_test_dt=DecisionClassifierScores(X, y)
#RandomizedSearch(X_train_dt, y_train_dt, clf_dt_model,X_test_dt, y_test_dt )
#clf_rf_model, X_train_rf, y_train_rf, X_test_rf, y_test_rf, y_pred_rf=RandomForestScores(X, y)
#RandomizedSearch(X_train_rf, y_train_rf, clf_rf_model, X_test_rf, y_test_rf)
#
clf_xg_model, X_train_xg, y_train_xg, X_test_xg, y_test_xg=XGBoostScores(X, y)
#createLernCurve(clf_xg_model, X, y)
#RandomizedSearch(X_train_xg, y_train_xg, clf_xg_model, X_test_xg, y_test_xg)
#createLernCurve(clf_dt_model, X, y)
createValidCurve(clf_xg_model, X, y, "Adult-Income")
#f1ScoreValidCurve(X, y)
