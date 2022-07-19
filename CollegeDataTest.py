#Nicholas Rudowski - 191963270 - Git: nickRudowski
#Sean Grecco - 170317470 - Git: SeanGrec
#Github: github.com/SeanGrec/CP468_FinalProject
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error,mean_squared_error

file = "usnewsDataNormalized.csv"
columns=['Fed ID #','College Name', 'State', 'public/private', 'Avg Math SAT Score', 'Avg Verb SAT Score', 'Avg Combined SAT Score', 'Avg ACT Score', 'First quartile - Math SAT', 'Third quartile - Math SAT', 'First quartile - Verbal SAT',
'Third quartile - Verbal SAT', 'First quartile - ACT', 'Third quartile - ACT', 'Number of applications received', 'Number of applicants accepted' , 'Number of new students enrolled',' Pct. new students from top 10%% of H.S. class', 
'Pct. new students from top 25%% of H.S. class', 'Number of fulltime undergraduates', 'Number of parttime undergraduates', 'In-state tuition', 'Out-of-state tuition', 'Room and board costs', 'Room costs', 'Board costs', 'Additional fees',
'Estimated book costs', 'Estimated personal spending', 'Pct. of faculty with Ph.D.\'s', 'Pct. of faculty with terminal degree', 'Student/faculty ratio', 'Pct.alumni who donate', 'Instructional expenditure per student', 'Graduation rate', 'Avg SAT %%', 'Avg ACT %%']
#Import and Read csv
df = pd.read_csv(file, names = columns, index_col="College Name")

#Include ACT scores if school doesn't have Combined SAT Scoore
#Normalize the data into a percentage (ie. x/36 or x/1600, -400 for SAT score)

df["Avg SAT %%"] = df["Avg SAT %%"].apply(pd.to_numeric, errors='coerce')
df["Avg SAT %%"].fillna(df["Avg ACT %%"], inplace=True)

#Start of SAT_Tuition Linear Regression
df_SATACTvT = df[["Avg SAT %%", "In-state tuition"]]
df_SATACTvT.columns = ["SAT", "Tuition"]
#Print Scores v Tuition info
#print(df_SATACTvT.info())

print("SAT/ACT v Tuition")

#Get rid of colleges with * in either SAT or Tuition columns.
df_SATACTvT = df_SATACTvT.apply (pd.to_numeric, errors='coerce')
df_SATACTvT = df_SATACTvT.dropna()
#print Scores v Tuition info
#print(df_SATACTvT.info())

sns.lmplot(x ="SAT", y ="Tuition", data = df_SATACTvT, order = 2, ci = None)

#Create numpy array of dataframe
X = np.array(df_SATACTvT['SAT']).reshape(-1, 1)
y = np.array(df_SATACTvT['Tuition']).reshape(-1, 1)


#Split data into training and testing -- 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Start linear regression test
regr = LinearRegression()
regr.fit(X_train, y_train)

#Output accuracy score of test
print("SCORE - SAT v T: ")
print(regr.score(X_test, y_test))

#Create graph
y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
plt.show()
#-----------------------------------------------------------
print("-----------------------------------------------------------")
#Start of SAT_GRAD Linear Regression
df_SATACT_GRAD = df[["Avg SAT %%", "Graduation rate"]]
df_SATACT_GRAD.columns = ["SAT", "Grad Rate"]

print("SAT/ACT v Grad")

#Get rid of colleges with * in either SAT or Grad Rate columns.
df_SATACT_GRAD = df_SATACT_GRAD.apply (pd.to_numeric, errors='coerce')
df_SATACT_GRAD = df_SATACT_GRAD.dropna()

#Print Scores and Grad info
#print(df_SATACT_GRAD.info())

#First graph
sns.lmplot(x ="SAT", y ="Grad Rate", data = df_SATACT_GRAD, order = 2, ci = None)

#Creating numpy arrays from dataframe
X = np.array(df_SATACT_GRAD['SAT']).reshape(-1, 1)
y = np.array(df_SATACT_GRAD['Grad Rate']).reshape(-1, 1)

#Split data into training and testing -- 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Start linear regression test
regr = LinearRegression()
regr.fit(X_train, y_train)

#Output accuracy score of test
print("SAT v Grad Score: ")
print(regr.score(X_test, y_test))

#Create graph
y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
plt.show()

#Create and output useful testing metric

mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
#Root mean squared error
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

#-----------------------------------------------------------
print("-----------------------------------------------------------")
print("Tuition v Grad")
#Start of Tuition_Grad Linear Regression
df_TvG = df[["In-state tuition", "Graduation rate"]]
df_TvG.columns = ["Tuition", "Grad Rate"]

#Print Tuition and Grad head
#print(df_TvG.head())

#Get rid of colleges with * in either SAT or Grad Rate columns.
df_TvG = df_TvG.apply (pd.to_numeric, errors='coerce')
df_TvG = df_TvG.dropna()

#First plot
sns.lmplot(x ="Tuition", y ="Grad Rate", data = df_TvG, order = 2, ci = None)

#Create numpy arrays from dataframe
X = np.array(df_TvG['Tuition']).reshape(-1, 1)
y = np.array(df_TvG['Grad Rate']).reshape(-1, 1)

#Splitting test and training data -- 75/25
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

#Starting linear regresson
regr = LinearRegression()
regr.fit(X_train, y_train)

#Output accuracy score
print("Tuition v Grad Score: ")
print(regr.score(X_test, y_test))

#Creat graph
y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
plt.show()

#Output useful testing metrics
mae = mean_absolute_error(y_true=y_test,y_pred=y_pred)
mse = mean_squared_error(y_true=y_test,y_pred=y_pred)
#Root mean squared error
rmse = mean_squared_error(y_true=y_test,y_pred=y_pred,squared=False)
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)

#-----------------------------------------------------------
#-----------------------------------------------------------
#-----------------------------------------------------------

#Kmeans time!
print("-----------------------------------------------------------")
print("KMEANS!")
#Create numpy array of Scores vs Tuition
X = np.array(df_SATACTvT)
#Print head for df
#print(df_SATACTvT.head())
#Create Kmeans with cluster of 10 (Represents 10%)
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
#Plot Kmeans
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
#Plot centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel("SAT Score")
plt.ylabel("Tuition")
plt.show()
#-----------------------------------------------------------

#Create numpy array for Scores vs Grad Rate
X = np.array(df_SATACT_GRAD)
#Print head of df
#print(df_SATACT_GRAD.head())
#Create KMeans with 10 clusters (Represents 10%)
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
#Plot Kmeans
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
#Plot Centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel("SAT Score")
plt.ylabel("Grad Rate")
plt.show()
#-----------------------------------------------------------

#Create numpy array for tuition vs grad rate
X = np.array(df_TvG)
#Print head for df
#print(df_TvG.head())
#Create Kmeans with 10 clusters (Represents 10%)
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)
#Plot KMeans
plt.scatter(X[:,0],X[:,1], c=kmeans.labels_, cmap='rainbow')
#Plot centroids
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.xlabel("Tuition")
plt.ylabel("Grad Rate")
plt.show()
