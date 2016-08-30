# coding=utf-8


from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.utils import check_random_state


data = load_svmlight_file("leu")
X_1 = data[0].todense().tolist()  # samples 72 features above 7129
y_1 = map(int,data[1])   # classes 2


#L1 SVM
l1svc = LinearSVC(penalty='l1', dual=False).fit(X_1, y_1)

coef = l1svc.coef_.tolist()[0]

#print(len(l1svc.coef_.tolist()[0]))

print("Number of features have non-zero weight vector coefficients %d " %sum(1 for i in coef if i  != 0))
