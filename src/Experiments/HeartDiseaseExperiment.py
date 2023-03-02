import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from src.Classifier.functions import DecisionClassifierScores, RandomForestScores, XGBoostScores, \
    RandomizedSearch, generateConfusionMatrix, createLernCurve, createValidCurve

df=pd.read_csv('/Users/yaseminakaydin/PycharmProjects/pythonProject/Classifier/Data/heart_cleveland_upload.csv', header=None)
df= df.iloc[1:]
print(df.dtypes)

df.columns = ['age',
              'sex',
              'cp',
              'trestbps',
              'chol',
              'fbs',
              'restecg'
             ,'thalach'
            ,'exang','oldpeak','slope','ca','thal','condition']

# Zählen der Anzahl von Instanzen pro Klasse
class_counts = df['condition'].value_counts()

# Ausgabe der Klassenverteilung
print(class_counts)

df=df._convert(numeric=True)
print(df.info)

#Split the data into dependent and independent Variables, X and Y
X = df.drop('condition', axis=1).copy()
y = df['condition'].copy()
class_col = 'condition'
class_distribution = df[class_col].value_counts()
class_distribution_percent = df[class_col].value_counts(normalize=True)
print("Klassenverteilung:")
print(class_distribution)
print("\nKlassenverteilung in Prozent:")
print(class_distribution_percent)

#One-Hot Encoding, wichtig für die Dokumentation, hier geht es um den Umgang
#von Attributen, welche eigentlich mehrere Kategorien vertreten

X_encoded=pd.get_dummies(X, columns=['cp',
                           'restecg',
                           'slope',
                           'thal'])


#clf_dt_model, X_train_dt, y_train_dt, X_test_dt, y_test_dt=DecisionClassifierScores(X, y)
#createLernCurve(clf_dt_model, X, y)
#RandomizedSearch(X_train_dt, y_train_dt, clf_dt_model,X_test_dt, y_test_dt )
#clf_rf_model, X_train_rf, y_train_rf, X_test_rf, y_test_rf, y_pred_rf=RandomForestScores(X, y)
#createLernCurve(clf_rf_model, X, y)
#RandomizedSearch(X_train_rf, y_train_rf, clf_rf_model, X_test_rf, y_test_rf)
clf_xg_model, X_train_xg, y_train_xg, X_test_xg, y_test_xg=XGBoostScores(X, y)
#createLernCurve(clf_xg_model, X, y)
RandomizedSearch(X_train_xg, y_train_xg, clf_xg_model, X_test_xg, y_test_xg)
#generateConfusionMatrix(clf_dt_model, X_test_dt, y_test_dt)

#createValidCurve(clf_rf_model, X, y,"Heart-Disease")











