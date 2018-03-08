from sklearn import tree
import graphviz
import numpy as np
import pandas as pd

titanic_df = pd.read_csv("./titanic_data_set/train.csv")
titanic_test = pd.read_csv("./titanic_data_set/test.csv")

# X = [[0, 0], [1, 1]]
# Y = [0, 1]
clf = tree.DecisionTreeClassifier()
# clf = clf.fit(X, Y)
# print(clf.predict([[2., 2.]]))
#
#
# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(iris.data, iris.target)
#
# dot_data = tree.export_graphviz(clf, out_file=None)
# graph = graphviz.Source(dot_data)
# graph.render("iris")