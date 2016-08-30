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
from scipy.sparse import csr_matrix
import csv
import random
from numpy import matrix
from scipy import sparse

#data
data = load_svmlight_file("leu")




# subSampling
l =len(data[1])
#print(map(int,data[1]))
start = int(round(l*0.70,0))

N = random.sample(range(start,l), 1)
#print(N)
i = np.random.choice(np.arange(data[0].shape[0]), N, replace=False)


sub_data = data[0][i.tolist()]
sub_sample = data[1][i.tolist()] # check for this step
#test_size= 0.4 # selecting number of samples


X_train, X_test, y_train, y_test = cross_validation.train_test_split(sub_data,sub_sample, test_size=0.4, random_state=1)

X_1 = X_train.todense().tolist()  # samples 72 features above 7129
y_1 = map(int,y_train)   # classes 2

#print(sub_data.shape)
X_1 = sub_data.todense().tolist()

#print(sparse.csr_matrix(X_1).shape)
y_1 = map(int,sub_sample)
#print(len(y_1))


#L2 SVM trained on the features selected by the L1 SVM subSampling
clf = LinearSVC(penalty='l1', dual=False).fit(X_1, y_1)
model = SelectFromModel(clf, prefit=True)
X = model.transform(X_1)

print("number of featuers selected %d",X.shape[1])
clf = LinearSVC(penalty='l2',dual=False)
scores = cross_validation.cross_val_score(clf, X, y_1, cv=10)

print(scores)
print("L2 SVM trained on the features selected by the L1 SVM subsampling. \n  Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))



#L2 SVM that uses RFE (with an L2-SVM) to select relevant features
clf = LinearSVC(penalty='l2',dual=False)
rfe = RFE(estimator=clf, n_features_to_select=10, step=1)
rfe.fit(data[0], data[1])
scores = cross_validation.cross_val_score(rfe.estimator_ , data[0], data[1], cv=10)
print(scores)
print("L2 SVM that uses RFE (with an L2-SVM) to select relevant features. \n Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


# featuer selection using The Golub score


def golub(data, labels):
    c = []
    for i in range(0,len(data)):
        c.append([labels[i],data[i]]) # append labels with data matrix

    negative = []
    positive = []

    for i in range(0,len(c)):
        if c[i][0] == [-1]:          # for negative condition
            negative.append(c[i][1])
        if c[i][0] == [1]:           # for positive condition
            positive.append(c[i][1])
    negMean = np.mean(np.asarray(negative),axis=0).tolist()  # calculate mean
    negStd = np.std(np.asarray(negative),axis=0).tolist()    # calculate standard deviation
    posMean = np.mean(np.asarray(positive),axis=0).tolist()  # calculating mean
    posStd = np.std(np.asarray(positive),axis=0).tolist()    # calculating Standard Deviation

    score =[]
    for i in range(0,len(negMean)):
        if((posStd[i]+negStd[i]) != 0):
            score.append((posMean[i]-negMean[i])/(posStd[i]+negStd[i]))
        else:
            score.append(0)    # zero if denominator is zero
    #print(len(score))
    return (score, score)  # return score as score , score as asked


#data = load_svmlight_file("leu")
X_1 = data[0].todense().tolist()
y_1 = map(int,data[1])
#print(len(X_1))
#print(len(X_1[0]))

lables= []
for i in range(0,len(y_1)):
    lables = lables + [[y_1[i]]]

#print X_1[0]
    #print(y_1)
    #print(lables)
filterData = np.zeros((len(y_1),1))
score = golub(X_1,lables)[0]
#print score
#print(len(X_1[1]))

filterData = filterData[:,:0]
#b = np.c_[[12, 13, 14, 15],b]

#b = np.c_[b,[ 0, 1, 2, 3]]
#print b
print len(score)

for i in range(0,len(score)):
    if(score[i] >0): # filter value
        #col = matrix(np.array(X_1)).transpose()[i].getA()[0]
        col = np.array(X_1)[:,i]
        filterData = np.c_[filterData,col]
        #filterData = np.append(filterData, col, 1)
#print sparse.csr_matrix(map(int,filterData)).shape

#dim = filterData.todense().tolist()
clf = LinearSVC(penalty='l2',dual=False)
#print data[0]
print("number of features(non zero golub score) selected %d", sparse.csr_matrix(filterData).shape[1])
#scores = cross_validation.cross_val_score(clf, filterData.reshape(len(dim),len(dim[0])), data[1].reshape(len(y_1),1), cv=10)
#scores = cross_validation.cross_val_score(clf, sparse.csr_matrix(filterData), sparse.csr_matrix(map(int,data[1])).reshape(len(y_1)), cv=10)
scores = cross_validation.cross_val_score(clf, sparse.csr_matrix(filterData), map(int,data[1]), cv=10)
#scores = cross_validation.cross_val_score(clf, filterData, data[1].reshape(len(y_1),1), cv=10)

print(scores)
print("L2 SVM trained on filtered features using golub score \n Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

