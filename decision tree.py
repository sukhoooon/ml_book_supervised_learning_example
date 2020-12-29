from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz
import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np

# 1) DT depth=5
cancer = load_breast_cancer()
X_2_train, X_2_test, Y_2_train, Y_2_test = train_test_split(cancer.data, cancer.target,
                                                            stratify=cancer.target, random_state=42)
tree = DecisionTreeClassifier(random_state=0)
tree.fit(X_2_train, Y_2_train)
print("accuracy of train set : {:.4f}".format(tree.score(X_2_train, Y_2_train)))
print("accuracy of test set : {:.4f}".format(tree.score(X_2_test, Y_2_test)))

tree = DecisionTreeClassifier(max_depth=5, random_state=0)
tree.fit(X_2_train, Y_2_train)
print("depth = 5, accuracy of train set : {:.4f}".format(tree.score(X_2_train, Y_2_train)))
print("depth = 5, accuracy of test set : {:.4f}".format(tree.score(X_2_test, Y_2_test)))

# 2) tree visualising - not working
# export_graphviz(tree, out_file="tree.dot", class_names = ["neg","pos"], feature_names = cancer.feature_names,
#                 impurity=False, filled=True)
# with open("tree.dot") as f:
#     dot_graph = f.read()
# dot = graphviz.Source(dot_graph)
# dot.format = 'pdf'
# dot.render(filename = 'tree.pdf')

# 3) feature priority
print("feature priority : {}".format(tree.feature_importances_))

def plot_feature_importances(model):
    n_feature = cancer.data.shape[1]
    plt.barh(range(n_feature),model.feature_importances_, align="center")
    plt.yticks(np.arange(n_feature), cancer.feature_names)
    plt.xlabel("feature importances")
    plt.ylabel("feature")

plot_feature_importances(tree)