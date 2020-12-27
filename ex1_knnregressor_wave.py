import mglearn
from mglearn import datasets
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X, Y = mglearn.datasets.make_wave(n_samples=40)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3)
reg.fit(X_train, Y_train)

fig, axes = plt.subplots(1, 3, figsize = (15, 4))
line = np.linspace(-3, 3, 1000).reshape(-1, 1)
for n_neighbors, ax in zip([1, 3, 9], axes):
    reg = KNeighborsRegressor(n_neighbors=n_neighbors)
    reg.fit(X_train, Y_train)
    ax.plot(line, reg.predict(line))
    ax.plot(X_train, Y_train, '^', c=mglearn.cm2(0), markersize=8)
    ax.plot(X_test, Y_test, 'v', c=mglearn.cm2(1), markersize=8)
    ax.set_title("{} training score = {:.2f} test score = {:.2f}".format(
        n_neighbors, reg.score(X_train, Y_train), reg.score(X_test, Y_test)))
    ax.set_xlabel('feature')
    ax.set_ylabel('target')
axes[0].legend(['model prediction', 'training data, target', 'test data, target'], loc='best')