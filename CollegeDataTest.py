#this is a test

import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans

file = "usnewsDataNormalized.csv"
columns=['Fed ID #','College Name', 'State', 'public/private', 'Avg Math SAT Score', 'Avg Verb SAT Score', 'Avg Combined SAT Score', 'Avg ACT Score', 'First quartile - Math SAT', 'Third quartile - Math SAT', 'First quartile - Verbal SAT',
'Third quartile - Verbal SAT', 'First quartile - ACT', 'Third quartile - ACT', 'Number of applications received', 'Number of applicants accepted' , 'Number of new students enrolled',' Pct. new students from top 10%% of H.S. class', 
'Pct. new students from top 25%% of H.S. class', 'Number of fulltime undergraduates', 'Number of parttime undergraduates', 'In-state tuition', 'Out-of-state tuition', 'Room and board costs', 'Room costs', 'Board costs', 'Additional fees',
'Estimated book costs', 'Estimated personal spending', 'Pct. of faculty with Ph.D.\'s', 'Pct. of faculty with terminal degree', 'Student/faculty ratio', 'Pct.alumni who donate', 'Instructional expenditure per student', 'Graduation rate', 'Avg SAT %%', 'Avg ACT %%']
df = pd.read_csv(file, names = columns, index_col="College Name")
print(df.head())

print(df[["Avg SAT %%", "Avg ACT %%"]])
df["Avg SAT %%"] = df["Avg SAT %%"].apply(pd.to_numeric, errors='coerce')
df["Avg SAT %%"].fillna(df["Avg ACT %%"], inplace=True)
print(df[["Avg SAT %%", "Avg ACT %%"]])

#Start of SAT_GRAD Linear Regression
df_SAT_GRAD = df[["Avg SAT %%", "Graduation rate"]]
df_SAT_GRAD.columns = ["SAT", "Grad Rate"]
print(df_SAT_GRAD.info())

print("SAT v Grad")

#Get rid of colleges with * in either SAT or Grad Rate columns.
df_SAT_GRAD = df_SAT_GRAD.apply (pd.to_numeric, errors='coerce')
df_SAT_GRAD = df_SAT_GRAD.dropna()

print(df_SAT_GRAD.info())

#Include ACT scores if school doesn't have Combined SAT Scoore
#Normalize the data into a percentage (ie. x/36 or x/1600, -400 for SAT score)

#plotting the Scatter plot to check relationship between Sal and Temp
sns.lmplot(x ="SAT", y ="Grad Rate", data = df_SAT_GRAD, order = 2, ci = None)


X = np.array(df_SAT_GRAD['SAT']).reshape(-1, 1)
y = np.array(df_SAT_GRAD['Grad Rate']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array
# since each dataframe contains only one column
df_SAT_GRAD.dropna(inplace = True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Splitting the data into training and testing data
regr = LinearRegression()

regr.fit(X_train, y_train)
print("SCORE1: ")
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
 
plt.show()
# Data scatter of predicted values
print("SAT 100 v Grad")

df_SAT_GRAD500 = df_SAT_GRAD[:][:100]
   
# Selecting the 1st 500 rows of the data
sns.lmplot(x ="SAT", y ="Grad Rate", data = df_SAT_GRAD500,
                               order = 2, ci = None)

df_SAT_GRAD500.fillna(method ='ffill', inplace = True)

X = np.array(df_SAT_GRAD500['SAT']).reshape(-1, 1)
y = np.array(df_SAT_GRAD500['Grad Rate']).reshape(-1, 1)

df_SAT_GRAD500.dropna(inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
print("SCORE2:")
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

#-----------------------------------------------------------

print("Tuition v Grad")
#Start of Tuition_Grad Linear Regression
df_TvG = df[["In-state tuition", "Graduation rate"]]
df_TvG.columns = ["Tuition", "Grad Rate"]
print(df_TvG.head())


#Get rid of colleges with * in either SAT or Grad Rate columns.
df_TvG = df_TvG.apply (pd.to_numeric, errors='coerce')
df_TvG = df_TvG.dropna()

print(df_TvG.head())

#Include ACT scores if school doesn't have Combined SAT Scoore
#Normalize the data into a percentage (ie. x/36 or x/1600, -400 for SAT score)

#plotting the Scatter plot to check relationship between Sal and Temp
sns.lmplot(x ="Tuition", y ="Grad Rate", data = df_TvG, order = 2, ci = None)


X = np.array(df_TvG['Tuition']).reshape(-1, 1)
y = np.array(df_TvG['Grad Rate']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array
# since each dataframe contains only one column
df_TvG.dropna(inplace = True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Splitting the data into training and testing data
regr = LinearRegression()

regr.fit(X_train, y_train)
print("SCORE3: ")
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
 
plt.show()
# Data scatter of predicted values

#Kmeans time!
print("KMEANS!")

X = np.array(df_SAT_GRAD)
print(df_SAT_GRAD.head())
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
plt.xlabel("SAT Score")
plt.ylabel("Grad Rate")
plt.show()

print("END")