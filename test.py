#this is a test

import math
import matplotlib.pyplot as plt
import pandas as pd

file = "usnews.data"
columns=['Fed ID #','College Name', 'State', 'public/private', 'Avg Math SAT Score', 'Avg Verb SAT Score', 'Avg Combined SAT Score', 'Avg ACT Score', 'First quartile - Math SAT', 'Third quartile - Math SAT', 'First quartile - Verbal SAT',
'Third quartile - Verbal SAT', 'First quartile - ACT', 'Third quartile - ACT', 'Number of applications received', 'Number of applicants accepted' , 'Number of new students enrolled',' Pct. new students from top 10%% of H.S. class', 
'Pct. new students from top 25%% of H.S. class', 'Number of fulltime undergraduates', 'Number of parttime undergraduates', 'In-state tuition', 'Out-of-state tuition', 'Room and board costs', 'Room costs', 'Board costs', 'Additional fees',
'Estimated book costs', 'Estimated personal spending', 'Pct. of faculty with Ph.D.\'s', 'Pct. of faculty with terminal degree', 'Student/faculty ratio', 'Pct.alumni who donate', 'Instructional expenditure per student', 'Graduation rate']
df = pd.read_csv(file, names = columns)
print(df.head())

print("Hello World")