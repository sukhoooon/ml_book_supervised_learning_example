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
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_moons

X, Y = make_moons(n_samples=100, noise=0.25, random_state=3)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, stratify=Y, random_state=42)
forest = RandomForestClassifier(n_estimators=5, random_state=2)
forest.fit(X_train, Y_train)

# plotting
fig, axes = plt.subplots(2, 3, figsize=(20, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), forest.estimators_)):
    ax.set_title("tree {}".format(i+1))
    mglearn.plots.plot_tree_partition(X, Y, tree, ax=ax)
mglearn.plots.plot_2d_separator(forest, X, fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("random forest")
mglearn.discrete_scatter(X[:, 0], X[:, 1], Y)

