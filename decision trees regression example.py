# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 09:15:40 2018
Decision tree regression example
@author: jahan
"""
import pandas as pd  
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
 
 
#import matplotlib.pyplot as plt  

dataset = pd.read_csv('C:/Users/jahan/Documents/SaintMaryCollege/Courses/OPS808/Summer 2018/PythonCodeExamples/Baseball_salary.csv')  

dataset.shape
sumNullRws = dataset.isnull().sum()
# remove null elements in data
dataset = dataset.dropna()
# check to see if there is any nulls left
dataset.isnull().sum()

# if no more nulls, we are ready to apply transformation
"""
df1.assign(e = pd.Series(np.random.randn(sLength), index=df1.index))
df1.loc[:,'f'] = p.Series(np.random.randn(sLength), index=df1.index)
"""
dataset.shape

array = np.log(dataset['Salary'].values)

#dataset1.assign(Salary = pd.Series(array))
dataset.loc[:,'SalaryLog'] = pd.Series(array, index=dataset.index)

dataset = dataset.dropna()

dataset.head(20)

dataset.describe()  

X = dataset.loc[:,['Hits', 'Years']]
#X = dataset.drop('Petrol_Consumption', axis=1)  
y = dataset['SalaryLog']  

 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0) 

  
regressor = DecisionTreeRegressor(max_depth=2)  
regressor.fit(X_train, y_train)  

y_pred = regressor.predict(X_test) 

df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
df  

  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))  


#import matplotlib.pyplot as plt

dot_data = StringIO()
export_graphviz(regressor, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  

Image(graph.create_png())


