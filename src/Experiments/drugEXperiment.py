import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from src.Classifier.functions import DecisionClassifierScores, RandomForestScores, XGBoostScores, createRandomGrid
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score



df = pd.read_csv('/Users/yaseminakaydin/PycharmProjects/pythonProject/Classifier/Data/drug200.csv')



encoder = LabelEncoder()
df['Sex'] = encoder.fit_transform(df['Sex'])
df['BP'] = encoder.fit_transform(df['BP'])
df['Cholesterol'] = encoder.fit_transform(df['Cholesterol'])
#df['Drug'] = encoder.fit_transform(df['Drug'])

print(df.head())
#print(df.dtypes)

df['Drug'].replace(['DrugY','drugC', 'drugX', 'drugA', 'drugB'],[0, 1,2,3,4],inplace=True)

#split the Data
X = df.drop('Drug', axis=1).copy()
y = df['Drug'].copy()
#print(df['Drug'].unique())

#tree, X_train, y_train =DecisionClassifierAccuracy(X, y)
tree, X_train, y_train, X_test, y_test, score=RandomForestScores(X, y)
print("Vorher" , score )
#XGBoostAccuracy(X, y)



#GridSearch

param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
    }

grid = GridSearchCV(tree, param_grid=param_grid, cv=10, verbose=1, n_jobs=-1)
grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_estimator_)
print(grid.best_score_)
rfc1=RandomForestClassifier(random_state=30, max_features='sqrt', n_estimators= 200, max_depth=6, criterion='gini')
rfc1.fit(X_train, y_train)
y_pred = rfc1.predict(X_test)
print(accuracy_score(y_test, y_pred))



