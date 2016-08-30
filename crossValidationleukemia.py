import numpy as np
from sklearn import cross_validation
from sklearn import svm
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import RFE
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
import matplotlib.pyplot as plt

#data
data = load_svmlight_file("leu")
#X_1 = data[0].todense().tolist()  # samples 72 features above 7192
#y_1 = map(int,data[1])   # classes 2

print(data[0].shape)
print(data[1].shape)

#test_size= 0.2 # selecting number of samples

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(data[0],data[1], test_size=0.2, random_state=1) # use it for subsampling
'''
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
'''
#L1 SVM
clf = LinearSVC(penalty='l1', dual=False)
scores = cross_validation.cross_val_score(clf, data[0], data[1], cv=10)

print(scores)
print("L1 SVM \n  Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#L2 SVM trained on all the features
clf = LinearSVC(penalty='l2',dual=False)
scores = cross_validation.cross_val_score(clf, data[0], data[1], cv=10)

print(scores)
print("L2 SVM trained on all the features \n Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


#L2 SVM trained on the features selected by the L1 SVM
clf = LinearSVC(penalty='l1', dual=False).fit(data[0], data[1])
model = SelectFromModel(clf, prefit=True)
X = model.transform(data[0])

print(X.shape)
clf = LinearSVC(penalty='l2',dual=False)
scores = cross_validation.cross_val_score(clf, X, data[1], cv=10)

print(scores)
print("L2 SVM trained on the features selected by the L1 SVM. \n  Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))




#L2 SVM that use the class RFECV which automatically selects the number of features

clf = LinearSVC(penalty='l2',dual=False)
# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=clf, step=1, cv=StratifiedKFold(data[1], 2),scoring='accuracy')
rfecv.fit(data[0], data[1])
#scores = cross_validation.cross_val_score(rfecv, data[0], data[1], cv=10)
print("Optimal number of features : %d" % rfecv.n_features_)
scores = rfecv.grid_scores_
print(scores)
print("L2 SVM that use the class RFECV which automatically selects the number of features. \n Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

'''
# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()
'''

#L2 SVM that uses RFE (with an L2-SVM) to select relevant features
clf = LinearSVC(penalty='l2',dual=False)
rfe = RFE(estimator=clf, n_features_to_select=10, step=1)
rfe.fit(data[0], data[1])
scores = cross_validation.cross_val_score(rfe.estimator_ , data[0], data[1], cv=10)
print(scores)
print("L2 SVM that uses RFE (with an L2-SVM) to select relevant features. \n Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

