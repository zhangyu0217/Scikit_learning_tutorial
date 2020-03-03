from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib
import graphviz
import sklearn

# Load dataset from the library
X, y = load_iris(return_X_y=True)
# Initiate a decision tree
clf = tree.DecisionTreeClassifier()
# give the tree a set of features and targets
clf = clf.fit(X, y)
# Optimize the decision tree
tree.plot_tree(clf.fit(X, y))
`
dot_data = tree.export_graphviz(clf, out_file=None) 
graph = graphviz.Source(dot_data) 
graph.render("iris")
