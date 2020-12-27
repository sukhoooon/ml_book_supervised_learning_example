import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

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