import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_boston, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
import numpy as np
from sklearn.svm import LinearSVC

# blobs example - (2 feature, 3 class) - multiclass
X, Y = make_blobs(random_state=42)
# scatter graph
mglearn.discrete_scatter(X[:,0], X[:,1], Y)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.legend(["class 1", "class 2", "class 3"])

#  linear svm boundary graph
linear_svm = LinearSVC().fit(X, Y)
print("shape of coeff : ", linear_svm.coef_.shape)
print("shape of intercept : ", linear_svm.intercept_.shape)

mglearn.discrete_scatter(X[:,0], X[:,1], Y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line*coef[0] + intercept) / coef[1], c = color)
plt.ylim(-10, 15)
plt.xlim(-10, 8)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.legend(["class 1", "class 2", "class 3", "boundary 1", "boundary2", "boundary3"], loc = 1)

# linear svm 2d graph
mglearn.plots.plot_2d_classification(linear_svm, X, fill=True, alpha = 0.7)
mglearn.discrete_scatter(X[:,0], X[:,1], Y)
line = np.linspace(-15, 15)
for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_,
                                  mglearn.cm3.colors):
    plt.plot(line, -(line*coef[0] + intercept) / coef[1], c = color)
plt.xlabel("feature1")
plt.ylabel("feature2")
plt.legend(["class 1", "class 2", "class 3", "boundary 1", "boundary2", "boundary3"], loc = 1)