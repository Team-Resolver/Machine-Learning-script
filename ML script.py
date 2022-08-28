""""""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
%matplotlib inline
# Libraries for data preparation and model building
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

#Different dataset from WHO and local health authorities
df = pd.read_csv("health.csv")
df.info
df.describe
len(df)
df.isnull().sum()/len(df)*100
df.duplicated().sum()
sns.barplot(data = df.Deaths)
df.dropna()
df.plot(kind='density', subplots=True,layout=(12,4) ,sharex=False, figsize=(20, 30))
df.hist(bins=20, figsize=(10,15))
df.kurtosis()
df.corr()
df.shape

# Modelling
X = df.iloc[: , :35]
Y = df.iloc[: , 35]
X_train, X_test, Y_train, Y_test= train_test_split(X,Y,test_size=0.20, random_state=1)
lm = LinearRegression()
lm.fit(X_train,Y_train)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
lm.score(X_test,Y_test)

