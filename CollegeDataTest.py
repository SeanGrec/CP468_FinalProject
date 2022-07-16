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

#Just SAT v Grad
print("SAT v Grad")

df_SAT_GRAD = df[['Avg SAT %%', "Graduation rate"]]
df_SAT_GRAD.columns = ["SAT", "Grad Rate"]
print(df_SAT_GRAD.info())
#Get rid of colleges with * in either SAT or Tuition columns.
df_SAT_GRAD = df_SAT_GRAD.apply (pd.to_numeric, errors='coerce')
df_SAT_GRAD = df_SAT_GRAD.dropna()

sns.lmplot(x ="SAT", y ="Grad Rate", data = df_SAT_GRAD,
                               order = 2, ci = None)

X = np.array(df_SAT_GRAD['SAT']).reshape(-1, 1)
y = np.array(df_SAT_GRAD['Grad Rate']).reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

regr = LinearRegression()
regr.fit(X_train, y_train)
print("SAT V Grad score:")
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')

plt.show()
#-----------------------------------------------------------


#Include ACT scores if school doesn't have Combined SAT Scoore
#Normalize the data into a percentage (ie. x/36 or x/1600, -400 for SAT score)
print(df[["Avg SAT %%", "Avg ACT %%"]])
df["Avg SAT %%"] = df["Avg SAT %%"].apply(pd.to_numeric, errors='coerce')
df["Avg SAT %%"].fillna(df["Avg ACT %%"], inplace=True)
print(df[["Avg SAT %%", "Avg ACT %%"]])

#Start of SAT_Tuition Linear Regression
df_SATACTvT = df[["Avg SAT %%", "In-state tuition"]]
df_SATACTvT.columns = ["SAT", "Tuition"]
print(df_SATACTvT.info())

print("SAT/ACT v Tuition")

#Get rid of colleges with * in either SAT or Tuition columns.
df_SATACTvT = df_SATACTvT.apply (pd.to_numeric, errors='coerce')
df_SATACTvT = df_SATACTvT.dropna()

print(df_SATACTvT.info())

X = np.array(df_SATACTvT['SAT']).reshape(-1, 1)
y = np.array(df_SATACTvT['Tuition']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array
# since each dataframe contains only one column
df_SATACTvT.dropna(inplace = True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Splitting the data into training and testing data
regr = LinearRegression()

regr.fit(X_train, y_train)
print("SCORE - SAT v T: ")
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
 
plt.show()

#-----------------------------------------------------------


#Start of SAT_GRAD Linear Regression
df_SATACT_GRAD = df[["Avg SAT %%", "Graduation rate"]]
df_SATACT_GRAD.columns = ["SAT", "Grad Rate"]
print(df_SATACT_GRAD.info())

print("SAT/ACT v Grad")

#Get rid of colleges with * in either SAT or Grad Rate columns.
df_SATACT_GRAD = df_SATACT_GRAD.apply (pd.to_numeric, errors='coerce')
df_SATACT_GRAD = df_SATACT_GRAD.dropna()

print(df_SATACT_GRAD.info())


#plotting the Scatter plot to check relationship between Sal and Temp
sns.lmplot(x ="SAT", y ="Grad Rate", data = df_SATACT_GRAD, order = 2, ci = None)


X = np.array(df_SATACT_GRAD['SAT']).reshape(-1, 1)
y = np.array(df_SATACT_GRAD['Grad Rate']).reshape(-1, 1)

# Separating the data into independent and dependent variables
# Converting each dataframe into a numpy array
# since each dataframe contains only one column
df_SATACT_GRAD.dropna(inplace = True)

# Dropping any rows with Nan values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

# Splitting the data into training and testing data
regr = LinearRegression()

regr.fit(X_train, y_train)
print("SAT v Grad Score: ")
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
 
plt.show()
# Data scatter of predicted values


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

print(df_TvG.info())

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
print("Tuition v Grad Score: ")
print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
 
plt.show()

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
#squared True returns MSE value, False returns RMSE value.
mse = mean_squared_error(y_true=y_test,y_pred=y_pred) #default=True
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)

print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)
# Data scatter of predicted values


#Kmeans time!
print("KMEANS!")

X = np.array(df_SATACTvT)
print(df_SATACTvT.head())
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel("SAT Score")
plt.ylabel("Tuition")
plt.show()


X = np.array(df_SATACT_GRAD)
print(df_SATACT_GRAD.head())
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel("SAT Score")
plt.ylabel("Grad Rate")
plt.show()

X = np.array(df_TvG)
print(df_TvG.head())
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel("Tuition")
plt.ylabel("Grad Rate")
plt.show()

print("END")