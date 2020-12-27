import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ex1) 산점도 example
X_1, Y_1 = mglearn.datasets.make_forge()

mglearn.discrete_scatter(X_1[:,0], X_1[:,1], Y_1)
plt.legend(["class 1", "class 2"], loc=1)
plt.title("ex1 point praph")
plt.xlabel("feature 1")
plt.ylabel("featur 2")
print("X_1.shape : {}".format(X_1.shape))

X_1_train, X_1_test, Y_1_train, Y_1_test = train_test_split(X_1, Y_1, random_state=0)
clf = KNeighborsClassifier(n_neighbors=3) # knn - 3 neighbor
clf.fit(X_1_train, Y_1_train)
print("테스트 데이터 세트 정확도: {:.2f}".format(clf.score(X_1_test, Y_1_test)))

fig, axes = plt.subplots(1, 3, figsize=(10, 3))
for n_neighbors, ax in zip([1, 3, 9], axes):
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_1, Y_1)
    mglearn.plots.plot_2d_separator(clf, X_1, fill=True, eps=0.5, ax=ax,
                                    alpha = .4)
    mglearn.discrete_scatter(X_1[:,0], X_1[:, 1], Y_1, ax=ax)
    ax.set_title("{} neighbors".format(n_neighbors))
    ax.set_xlabel("feature 1")
    ax.set_ylabel("featur 2")
axes[0].legend(loc=3)



## ex2) 유방암 데이터셋 (악성212, 양성357)
cancer = load_breast_cancer()
print("cancer.keys(): \n{}".format(cancer.keys()))
print("유방암 데이터 형태 : {}".format(cancer.data.shape))

X_2_train, X_2_test, Y_2_train, Y_2_test = train_test_split(cancer.data, cancer.target,
                                                            stratify=cancer.target, random_state=66)
training_accuracy = []
test_accuracy = []
neighbors_settings = range(1,11)

for n_neighbors in neighbors_settings:
    clf = KNeighborsClassifier(n_neighbors = n_neighbors)
    clf.fit(X_2_train, Y_2_train)
    training_accuracy.append(clf.score(X_2_train, Y_2_train))
    test_accuracy.append(clf.score(X_2_test, Y_2_test))

plt.plot(neighbors_settings, training_accuracy, label='accuracy of training data')
plt.plot(neighbors_settings, test_accuracy, label='accuracy of test data')
plt.ylabel('accuracy')
plt.xlabel('n_neighbors')
plt.show()


## ex3) 보스턴 주택 가격 (506 데이터, 13 특성)
boston = load_boston()
print("보스턴 집값 데이터 형태 :{}".format(boston.data.shape))
X_3, Y_3 = mglearn.datasets.load_extended_boston() # 특성들간의 결합 : extended feature (feature engineering - 4장)
print("X_3.shape extended feature : {}".format(X_3.shape))
