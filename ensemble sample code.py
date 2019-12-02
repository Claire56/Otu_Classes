# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 18:33:39 2017

@author:
"""

# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
#import os

# to make this notebook's output stable across runs
np.random.seed(42)

# To plot pretty figures
#%matplotlib inline
#import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

## Where to save the figures
#PROJECT_ROOT_DIR = "C:/Users/jahan/Documents/SaintMaryCollege/Courses/OPS808/Summer 2018/PythonCodeExamples/figures"
##CHAPTER_ID = "decision_trees"
#
#def image_path(fig_id):
#    #return os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id)
#    return os.path.join(PROJECT_ROOT_DIR, fig_id)
#
#def save_fig(fig_id, tight_layout=True):
#    print("Saving figure", fig_id)
#    if tight_layout:
#        plt.tight_layout()
#    plt.savefig(image_path(fig_id) + ".png", format='png', dpi=300)


# load moon data and split into training and test sets    
from sklearn.model_selection import train_test_split
#from sklearn.datasets import make_moons
import pandas as pd  

"""
In this section we will predict whether a bank note is authentic or fake 
depending upon the four different attributes of the image of the note. 
The attributes are Variance of wavelet transformed image, 
Kurtosis of the image, entropy, and skewness of the image.
"""
dir_path = 'C:/Users/jg/Documents/MLData/'
filename = 'bill_authentication.csv'
dataset = pd.read_csv(dir_path + filename)  # to point to data


dataset.shape

dataset.head(20)

X = dataset.drop('Class', axis=1)  
y = dataset['Class']  

"""
Preparing the Data
In this section we will divide our data into attributes and labels 
and will then divide the resultant data into both training and test sets. 
By doing this we can train our algorithm on one set of data and then test it out 
on a completely different set of data that the algorithm hasn't seen yet. 
This provides you with a more accurate view of how your trained 
algorithm will actually perform. We now split the data
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  

 #   Bagging ensembles

from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    max_samples=100, bootstrap=True, n_jobs=-1, random_state=42)
bag_clf.fit(X_train, y_train)
y_pred = bag_clf.predict(X_test)

# determine accuracy score for the bagging method
print(accuracy_score(y_test, y_pred))

# now use a standard decision tree classifier
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)

print(accuracy_score(y_test, y_pred))
# compare bagging method with standard decision tree classifier


# Random Forests

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve

rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1, random_state=42)
rnd_clf.fit(X_train, y_train)
y_prob_rf = rnd_clf.predict_proba(X_test)

y_pred_rf = rnd_clf.predict(X_test)

np.sum(y_pred == y_pred_rf) / len(y_pred)  # almost identical predictions

y_score_rf = y_prob_rf[:,1]
fpr_rf,tpr_rf, threshold_rf = roc_curve(y_test, y_score_rf)

def plot_roc_curve(fpr, tpr, lable=None):
    plt.plot(fpr,tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
plot_roc_curve(fpr_rf,tpr_rf)
plt.legend(loc="bottom right")
plt.show()

# Out-of-Bag evaluation

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(random_state=42), n_estimators=500,
    bootstrap=True, n_jobs=-1, oob_score=True, random_state=40)
bag_clf.fit(X_train, y_train)
bag_clf.oob_score_

from sklearn.metrics import accuracy_score
y_pred = bag_clf.predict(X_test)
accuracy_score(y_test, y_pred)



# Boosting method, three boosting algorithms
# First Ada boost
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)

y_pred_ada = ada_clf.predict(X_test)
accuracy_score(y_test, y_pred_ada)
