from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib
import graphviz
import sklearn

import numpy as np

matplotlib.use("TkAgg")

# Load dataset from the library
X, y = load_iris(return_X_y=True)
print(X)
print(y)
iris = load_iris()
print(iris)
# Initiate a decision tree
clf = tree.DecisionTreeClassifier()
# give the tree a set of features and targets
clf = clf.fit(X, y)
# Optimize the decision tree
tree.plot_tree(clf.fit(X, y))

#dot_data = tree.export_graphviz(clf, out_file=None, filled=True) 
#graph = graphviz.Source(dot_data) 
#graph.render("iris")

#Task 1 : generate training and test datasets
# 0 : background 1 : signal
# signal :  x1 = Gaus(0,1), x2 = Gaus(100,50)
# bkg :     x1 = Gaus(5,3), x2 = Exp(50)
dataset = {}
data = []
target = []

for i in range(0,1000):
    x1_signal = float(np.random.normal(0,1))
    x2_signal = float(np.random.normal(100,500))
    data.append([x1_signal,x2_signal])
    target.append(1)

for i in range(0,1000):
    x1_bkg = float(np.random.normal(5,3))
    x2_bkg = float(np.random.exponential(50))
    data.append([x1_bkg, x2_bkg])
    target.append(0)

print(data)
print(target)

#Task 2 : train
#Task 2.1 : perform the training
#Task 2.2 : check the over training
#Task 2.3 : compare different configuration



#Task 3 : output the structure
