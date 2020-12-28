import mglearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Ridge, Lasso
import numpy as np

# ex3) 보스턴 주택 가격 (506 데이터, 13 특성)
boston = load_boston()
print("보스턴 집값 데이터 형태 :{}".format(boston.data.shape))
X_3, Y_3 = mglearn.datasets.load_extended_boston() # 특성들간의 결합 : extended feature (feature engineering - 4장)
print("X extended feature : {}".format(X_3.shape))

# linear regression
X_train, X_test, Y_train, Y_test = train_test_split(X_3, Y_3, random_state=0)
lr = LinearRegression().fit(X_train, Y_train)
print('X train set shape : {}'.format(X_train.shape))
print('Y train set shape : {}'.format(Y_train.shape))
print('lr train set score : {:,.2f}'.format(lr.score(X_train, Y_train)))
print('lr test set score : {:,.2f}'.format(lr.score(X_test, Y_test))) # result : overfitting

# ridge : alpha 조절 가능
ridge = Ridge().fit(X_train, Y_train)
print('ridge train set score : {:,.2f}'.format(ridge.score(X_train, Y_train)))
print('ridge test set score : {:,.2f}'.format(ridge.score(X_test, Y_test)))

ridge10 = Ridge(alpha = 10).fit(X_train, Y_train)
print('ridge10 train set score : {:,.2f}'.format(ridge10.score(X_train, Y_train)))
print('ridge10 test set score : {:,.2f}'.format(ridge10.score(X_test, Y_test)))

ridge01 = Ridge(alpha = 0.1).fit(X_train, Y_train)
print('ridge01 train set score : {:,.2f}'.format(ridge01.score(X_train, Y_train)))
print('ridge01 test set score : {:,.2f}'.format(ridge01.score(X_test, Y_test)))

plt.plot(ridge10.coef_, '^', label="Ridge alpha = 10")
plt.plot(ridge.coef_, 's', label="Ridge alpha = 1")
plt.plot(ridge01.coef_, 'v', label="Ridge alpha = 0.1")

plt.plot(lr.coef_, 'o', label="Linear regression")
plt.xlabel("coeff list")
plt.ylabel("coeff magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-30, 30)
plt.legend()

# sample number dependency - learning curve
# mglearn.plots.plot_ridge_n_samples()

#  Lasso
lasso = Lasso().fit(X_train, Y_train)
print('Lasso train set score : {:,.2f}'.format(lasso.score(X_train, Y_train)))
print('Lasso test set score : {:,.2f}'.format(lasso.score(X_test, Y_test)))
print('number of used feature : {}'.format(np.sum(lasso.coef_ !=0)))




