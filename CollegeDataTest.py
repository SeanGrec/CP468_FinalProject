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
print(df_SAT_GRAD.head())



'''
X = df["Avg Combined SAT Score"].to_numpy()
kmeans = KMeans(n_clusters=2).fit(X)
'''

print("Hello World")