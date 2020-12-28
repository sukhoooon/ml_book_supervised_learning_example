import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
import numpy as np
from sklearn.svm import LinearSVC

# 1) forge data - svc, logistic regression (2-D)
X, Y = mglearn.datasets.make_forge()
fig, axes = plt.subplots(1, 2, figsize = (10, 3))
for model, ax in zip([LinearSVC(), LogisticRegression()], axes):
    clf = model.fit(X, Y)
    mglearn.plots.plot_2d_separator(clf, X, fill=False, eps=0.5,
                                    ax=ax, alpha=.7)
    mglearn.discrete_scatter(X[:, 0], X[:,1], Y, ax=ax)
    ax.set_title("{}".format(clf.__class__.__name__))
    ax.set_xlabel("featuer1")
    ax.set_ylabel("feature2")
axes[0].legend()

# dependency on c
mglearn.plots.plot_linear_svc_regularization()

# 2) breast cancer data
cancer = load_breast_cancer()
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target,
                                                            stratify=cancer.target, random_state=42)
logreg = LogisticRegression().fit(X_train, Y_train)
print("score of train set : {:.3f}".format(logreg.score(X_train, Y_train)))
print("score of test set : {:.3f}".format(logreg.score(X_test, Y_test))) # good but under~

logreg100 = LogisticRegression(C=100).fit(X_train, Y_train)
print("score of train set : {:.3f}".format(logreg100.score(X_train, Y_train)))
print("score of test set : {:.3f}".format(logreg100.score(X_test, Y_test)))

logreg001 = LogisticRegression(C=0.01).fit(X_train, Y_train)
print("score of train set : {:.3f}".format(logreg001.score(X_train, Y_train)))
print("score of test set : {:.3f}".format(logreg001.score(X_test, Y_test)))

plt.plot(logreg.coef_.T, 'o', label="C=1")
plt.plot(logreg100.coef_.T, '^', label="C=100")
plt.plot(logreg001.coef_.T, 'v', label="C=0.01")
plt.xticks(range(cancer.data.shape[1]), cancer.feature_names, rotation=90)
plt.hlines(0, 0, cancer.data.shape[1])
plt.ylim([-5, 5])
plt.xlabel("feature")
plt.ylabel("magnitude of coefficients")
plt.legend()




