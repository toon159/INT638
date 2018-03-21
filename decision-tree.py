from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
# %matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import train_test_split


titanic_df = pd.read_csv("./titanic_data_set/train.csv", index_col=0)
titanic_test = pd.read_csv("./titanic_data_set/test.csv")

print ("Dataset Lenght:: ", len(titanic_df))
print ("Dataset Shape:: ", titanic_df.shape)
print(titanic_df.columns)

titanic_df = titanic_df[['Survived', 'Pclass', 'SibSp', 'Parch']]

print(titanic_df.head())

X_train = titanic_df.loc[:, 'Pclass':'Parch']
Y_train = titanic_df.loc[:, 'Survived']

dtree = DecisionTreeClassifier(max_leaf_nodes=3, random_state=0)
dtree.fit(X_train, Y_train)

print(dtree.predict([[1,1,0]]))
#
# X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.3, random_state = 100)
# The above snippet will split data into training and test set.
# X_train, y_train are training data &  X_test, y_test belongs to the test dataset.

# The parameter test_size is given value 0.3; it means test sets will be 30% of whole dataset
# & training dataset’s size will be 70% of the entire dataset.
# random_state variable is a pseudo-random number generator state used for random sampling.
# If you want to replicate our results, then use the same value of random_state.
#
#
# clf_entropy = DecisionTreeClassifier(criterion = "entropy", random_state = 100,
#  max_depth=3, min_samples_leaf=5)
# print(clf_entropy.fit(X_train, y_train))
# print(df)

dot_data = dtree.export_graphviz(X_train, out_file='dtree.dot')
Image(graph.create_png())
