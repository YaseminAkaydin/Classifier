import pandas as pd
import matplotlib  as plt
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, validation_curve, \
    cross_val_score
from sklearn import metrics
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
from scipy.stats import randint
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve





def featureImportanceCalculater(X, model):
    feature_list = list(X.columns)
    feauture_imp = pd.Series(model.feature_importances_, index=feature_list).sort_values(ascending=False)
    return  feauture_imp


def DecisionClassifierScores(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
    clf_dt = tree.DecisionTreeClassifier(random_state=42)
    clf_dt = clf_dt.fit(X_train, y_train)
    y_pred = clf_dt.predict(X_test)
    print("---Decision Tree---")
    print("Accuracy: " , accuracy_score(y_test, y_pred))
    print("Train-Score: ", clf_dt.score(X_train, y_train))
    print("F1-Score: ", f1_score(y_test, y_pred))
    print("ROC-AUC-Score: ", roc_auc_score(y_test, y_pred))
    print("---Parameter-Values")
    print("Max_depth", clf_dt.get_depth())
    print("Leaves", clf_dt.get_n_leaves())

    print(clf_dt.get_params())



    return clf_dt, X_train, y_train, X_test, y_test

def RandomForestScores(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    clf_rf = RandomForestClassifier(random_state=30)
    clf_rf.fit(X_train, y_train)
    y_pred = clf_rf.predict(X_test)
    print("---Random Forest---")
    print("Accuracy: ", accuracy_score(y_test, y_pred))
    print("Train-Score: ", clf_rf.score(X_train, y_train))
    print("F1-Score: ", f1_score(y_test, y_pred))
    print("ROC-AUC-Score: ", roc_auc_score(y_test, y_pred))
    return clf_rf, X_train, y_train, X_test, y_test, y_pred

def XGBoostScores(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y)
    #print(sum(y_train) / len(y_train))

    clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42)
    clf_xgb.fit(X_train, y_train, verbose=True, early_stopping_rounds=10, eval_metric='aucpr',
                eval_set=[(X_test, y_test)])
    y_pred = clf_xgb.predict(X_test)
    print("---XGBoost---")
    print("Accuracy: ", accuracy_score(y_test, y_pred) )
    print("Train-Score: ", clf_xgb.score(X_train, y_train))
    print("F1-Score: ", f1_score(y_test, y_pred))
    print("ROC-AUC-Score: ", roc_auc_score(y_test, y_pred))

    return clf_xgb, X_train, y_train, X_test, y_test


def createRandomGrid(n):
    if(n==1):
        random_grid = {
            'n_estimators': randint(18, 75),#(20, 150)
            'max_depth': randint(10, 25),#(5, 20)
            'min_samples_split': randint(55, 120),#randint(25, 120)
            'min_samples_leaf': randint(20, 35),#randint(2, 25)
            'max_features': ['sqrt', 'log2']
        }


    elif (n==0):
        random_grid = {
                'criterion': ['gini', 'entropy'],
                'max_depth': randint(2, 50),
                'min_samples_split': randint(2, 50), # [2, 5, 10]
                'min_samples_leaf': randint(2, 100)
            }

    elif (n==2):
        random_grid= {
            'n_estimators': randint(50, 70),#[100, 500, 1000, 2000]
            'learning_rate': [ 0.2, 0.3, 0.5, 1.7, 1.0],#[0.01, 0.05, 0.1, 0.3, 0.5]
            'max_depth': randint(2, 20),#[3, 5, 7, 9]
            'subsample': [ 0.6,  0.7, 0.8, 0.9],#[0.5, 0.7, 1.0]
            'colsample_bytree': [ 0.6, 0.9, 1.0 ],#[0.5, 0.7, 1.0]
            'gamma': [ 0, 0.1, 0.5, 1, 1.5, 1.6, 1.7, 1.8, 2.0],#[0, 0.1, 0.5, 1, 1.5]
            'reg_alpha': [ 0.01, 0.1, 0.2, 0.15],#[0, 0.01, 0.1, 1, 10]
            'reg_lambda': [ 0.1, 1, 10, 12],#[0, 0.01, 0.1, 1, 10]

        }
    return random_grid

def RandomizedSearch(X_train, y_train, model, X_test, y_test):
    param_grid= createRandomGrid(2)
    randomGrid= RandomizedSearchCV(estimator=model, param_distributions=param_grid, scoring="accuracy",  n_iter = 100, cv = 5, verbose=2, random_state=35, n_jobs = -1)
    randomGrid.fit(X_train,y_train)
    best_model = randomGrid.best_estimator_
    best_parameters= randomGrid.best_params_
    best_model.fit(X_train, y_train)
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    #grid = GridSearchCV(model, param_grid=param_grid, scoring='accuracy', cv=10, verbose=1, n_jobs=-1)
    #grid.fit(X_train, y_train)
    print("---after Hyperparameter Tuning")
    print("Neue Accuracy: ", accuracy)
    print(best_parameters)
    print("Fr-Score", f1_score(y_test, y_pred))
    print("Roc-Auc-SCore", roc_auc_score(y_test, y_pred))
    print("Tiefe des Baumes", best_parameters)

    # print("Best Params: ", randomGrid.best_params_)
    # print("Best Estimator: ", randomGrid.best_estimator_)
    # print("Best Score: ", randomGrid.best_score_)
    print("Train-Accuracy Nachher: ", best_model.score(X_train, y_train))






def generateConfusionMatrix(model, X_test, y_test ):
    # Berechnen der Konfusionsmatrix
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    # Zeichnen der Heatmap
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')

    # Speichern der Konfusionsmatrix als Bild
    plt.show()

def createLernCurve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label="Trainingsgenauigkeit")
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label="Testgenauigkeit")
    plt.legend()
    plt.show()

def createValidCurve(model, X, y,  dataset):
    max_depth_range = randint(1, 100)

    # Verwenden Sie die validation_curve-Funktion von Scikit-Learn, um die Trainings- und Testgenauigkeit
    # des Klassifikators für jeden Wert von max_depth zu berechnen
    train_scores, test_scores = validation_curve(
        model, X, y, param_name="max_depth", param_range=max_depth_range,
        cv=5, scoring="f1", n_jobs=-1)

    # Berechne den Mittelwert und die Standardabweichung der Genauigkeit für jeden Wert von max_depth
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)


    plt.title(f"Validation Curve {dataset}")
    plt.xlabel("max_depth")
    plt.ylabel("F1-Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(max_depth_range, train_scores_mean, label="Training score",
             color="darkorange", lw=lw)
    plt.fill_between(max_depth_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(max_depth_range, test_scores_mean, label="Cross-validation score",
             color="navy", lw=lw)
    plt.fill_between(max_depth_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()

def f1ScoreValidCurve( X, y):
    max_depths = [1, 5, 10, 15, 20, 30, 50, 100]

    # Liste für den Durchschnittswert und Standardabweichung des F1-Scores pro max_depth
    f1_scores_mean = []
    f1_scores_std = []

    # Durchlaufe jeden max_depth-Wert und berechne den F1-Score
    for depth in max_depths:
        clf = xgb.XGBClassifier(max_depth=depth)
        f1_scores = cross_val_score(clf, X, y, cv=5, scoring='f1')
        f1_scores_mean.append(np.mean(f1_scores))
        f1_scores_std.append(np.std(f1_scores))

    # Plotten der Validierungskurve
    plt.errorbar(max_depths, f1_scores_mean, yerr=f1_scores_std, fmt='-o', capsize=5)
    plt.xlabel('max_depth')
    plt.ylabel('F1-Score')
    plt.title('Validierungskurve')
    plt.show()






