import pandas as pd
import matplotlib  as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.Classifier.functions import RandomForestScores

df= pd.read_csv("/Users/yaseminakaydin/PycharmProjects/pythonProject/Classifier/Data/Iris.csv")
#print(df.head(20))

sizes= df['Species'].value_counts(sort=1)
print(sizes)

#Drop irrelevant columns
df.drop(['Id'], axis=1, inplace=True)
#print(df.head(5))

#Handle missing values
df = df.dropna()

#Convert non-numeric data to numeric
df.Species[df.Species == 'Iris-setosa']=0
df.Species[df.Species == 'Iris-versicolor']=1
df.Species[df.Species == 'Iris-virginica']=2
print(df.head(100))

#Define dependent variables
#what we are trying to predict
y = df['Species'].values
y = y.astype('int')

#Define independent variables (our features)
X= df.drop(labels=['Species'], axis=1)

#Split data into train and test datasets
RandomForestScores(X, y)




