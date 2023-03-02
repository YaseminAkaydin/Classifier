import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
import xgboost as xgb
from  sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, f1_score

from src.Classifier.functions import XGBoostScores, RandomForestScores, DecisionClassifierScores, \
     RandomizedSearch, createValidCurve, createLernCurve, generateConfusionMatrix

#import the data
df= pd.read_csv('/Users/yaseminakaydin/PycharmProjects/pythonProject/Classifier/Data/WA_Fn-UseC_-Telco-Customer-Churn 2.csv')

df=df._convert(numeric=True)
df.drop('customerID', axis=1)
del df['customerID']
df['gender'].replace(['Female','Male'],[0, 1],inplace=True)
df['Partner'].replace(['Yes','No'],[0, 1],inplace=True)
df['Dependents'].replace(['Yes','No'],[0, 1],inplace=True)
df['PhoneService'].replace(['Yes','No'],[0, 1],inplace=True)
df['MultipleLines'].replace(['Yes','No', 'No phone service'],[0, 1, 2],inplace=True)
df['InternetService'].replace(['DSL','Fiber optic','No'],[0, 1, 2],inplace=True)
df['OnlineSecurity'].replace(['Yes','No', 'No internet service'],[0, 1, 2],inplace=True)
df['OnlineBackup'].replace(['Yes','No','No internet service'],[0, 1, 2],inplace=True)
df['DeviceProtection'].replace(['Yes','No','No internet service' ],[0, 1, 2],inplace=True)
df['TechSupport'].replace(['Yes','No','No internet service'],[0, 1, 2],inplace=True)
df['StreamingTV'].replace(['Yes','No','No internet service'],[0, 1, 2],inplace=True)
df['StreamingMovies'].replace(['Yes','No','No internet service'],[0, 1, 2],inplace=True)
df['Contract'].replace(['Month-to-month','One year','Two year'],[0, 1, 2],inplace=True)
df['PaperlessBilling'].replace(['Yes','No'],[0, 1],inplace=True)
df['PaymentMethod'].replace(['Electronic check','Mailed check','Credit card (automatic)', 'Bank transfer (automatic)' ],[0, 1, 2, 3],inplace=True)
df['Churn'].replace(['Yes','No'],[0, 1],inplace=True)

#print(df.head())
#print(df.dtypes)
print(df.info)
print(df.size)


#Split the Data into dependent and independent variables
X = df.drop('Churn', axis=1).copy()
y = df['Churn'].copy()


#One-Hot Encoding
#pd.get_dummies(X, columns=['PaymentMethod']) -> eig alle Categricola columns

#Build a Preliminary XGBoost Model
#print(sum(y)/len(y))
#XGBoostAccuracy(X, y)
#
X["TotalCharges"] = X["TotalCharges"].replace(np.nan, 0)
#print(X[X.isna().any(axis=1)])
# Zählen der Anzahl von Instanzen pro Klasse
class_counts = df['Churn'].value_counts()

# Ausgabe der Klassenverteilung
print(class_counts)
# tree, X_train, y_train, X_test, y_test, y_pred=RandomForestAccuracy(X, y)
# print("Train-Accuracy before: ", tree.score(X_train, y_train))
# print("Test-Accuracy before: ", tree.score(X_test, y_test))
# print("Roc-Auc-Score", tree.roc_auc_score(X_test, y_test))
#DecisionClassifierAccuracy(X, y)

class_col = 'Churn'
class_distribution = df[class_col].value_counts()
class_distribution_percent = df[class_col].value_counts(normalize=True)
print("Klassenverteilung:")
print(class_distribution)
print("\nKlassenverteilung in Prozent:")
print(class_distribution_percent)


#Confusion Matrix erstellen
#print(confusion_matrix(y_test, y_pred))

#entsprechende Raten der Confusion Matrix
#cm = confusion_matrix(y_test, y_pred)
#cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#print(cm_norm)

#Precision-Recall
#traget_names= ['0', '1']
#print(classification_report(y_test, y_pred, digits=3, target_names=traget_names))
#print("F1-Score", f1_score(y_test, y_pred))

#ROC-Kurve
#print(roc_auc_score(y_test, tree.predict_proba(X_test)[:,1]))




clf_dt_model, X_train_dt, y_train_dt, X_test_dt, y_test_dt=DecisionClassifierScores(X, y)
#RandomizedSearch(X_train_dt, y_train_dt, clf_dt_model,X_test_dt, y_test_dt )
#clf_rf_model, X_train_rf, y_train_rf, X_test_rf, y_test_rf, y_pred_rf=RandomForestScores(X, y)
#RandomizedSearch(X_train_rf, y_train_rf, clf_rf_model, X_test_rf, y_test_rf)
clf_xg_model, X_train_xg, y_train_xg, X_test_xg, y_test_xg=XGBoostScores(X, y)
RandomizedSearch(X_train_xg, y_train_xg, clf_xg_model, X_test_xg, y_test_xg)
#createLernCurve(clf_xg_model, X, y)
#generateConfusionMatrix(clf_dt_model, X_test_dt, y_test_dt)
#createValidCurve(clf_xg_model, X, y, "Telco-Churn")







