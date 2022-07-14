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

file = "usnews.data"
columns=['Fed ID #','College Name', 'State', 'public/private', 'Avg Math SAT Score', 'Avg Verb SAT Score', 'Avg Combined SAT Score', 'Avg ACT Score', 'First quartile - Math SAT', 'Third quartile - Math SAT', 'First quartile - Verbal SAT',
'Third quartile - Verbal SAT', 'First quartile - ACT', 'Third quartile - ACT', 'Number of applications received', 'Number of applicants accepted' , 'Number of new students enrolled',' Pct. new students from top 10%% of H.S. class', 
'Pct. new students from top 25%% of H.S. class', 'Number of fulltime undergraduates', 'Number of parttime undergraduates', 'In-state tuition', 'Out-of-state tuition', 'Room and board costs', 'Room costs', 'Board costs', 'Additional fees',
'Estimated book costs', 'Estimated personal spending', 'Pct. of faculty with Ph.D.\'s', 'Pct. of faculty with terminal degree', 'Student/faculty ratio', 'Pct.alumni who donate', 'Instructional expenditure per student', 'Graduation rate']
df = pd.read_csv(file, names = columns, index_col="College Name")
print(df.head())

df_SAT_GRAD = df[["Avg Combined SAT Score", "Graduation rate"]]
df_SAT_GRAD.columns = ["SAT", "Grad Rate"]
print(df_SAT_GRAD.head())
#Get rid of colleges with * in either SAT or Grad Rate columns.
df_SAT_GRAD = df_SAT_GRAD.apply (pd.to_numeric, errors='coerce')
df_SAT_GRAD = df_SAT_GRAD.dropna()

#If grad rate is over 100%, set it to 100%
for x in df_SAT_GRAD.index:
  if df_SAT_GRAD.loc[x, "Grad Rate"] > 100:
    df_SAT_GRAD.drop(x, inplace = True)

print(df_SAT_GRAD.head())

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

print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)
plt.scatter(X_test, y_test, color ='b')
plt.plot(X_test, y_pred, color ='k')
 
plt.show()
# Data scatter of predicted values



'''
X = df["Avg Combined SAT Score"].to_numpy()
kmeans = KMeans(n_clusters=2).fit(X)
'''

print("End")