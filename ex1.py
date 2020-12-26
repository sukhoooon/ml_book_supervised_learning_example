import mglearn
import matplotlib.pyplot as plt

# dataset
X, Y = mglearn.datasets.make_forge()

# 산점도 graph
mglearn.discrete_scatter(X[:,0], X[:,1], Y)
plt.legend(["클래스1", "클래스2"], loc=1)
plt.xlabel("첫번째 특성")
plt.ylabel("두번째 특성")
print("X.shape : {}".format(X.shape))
