# coding=utf-8

import numpy as np
from sklearn.svm import LinearSVC
from sklearn.datasets import load_svmlight_file
from sklearn.utils import check_random_state
from scipy.sparse import csr_matrix
import csv

f = open("arcene_train_valid.data") # read the data file
f1 = open("arcene_train_valid.labels") # read the lable file
try:
    a = []
    for row in csv.reader(f):
        row = [int(i) for i in row[0].split()]
        a.append(row)   # data matrix
    b = []
    for row in csv.reader(f1):
        row = map(int,row)
        b.append(row[0])   # labels matrix
        #print(golub(a,b))
finally:
    f.close   # close the files
    f1.close

a =csr_matrix(a) # convert into sparse matrix
b = np.asarray(b)


data =  (a,b)
X_1 = data[0].todense().tolist() 
y_1 = map(int,data[1])   

print(a.shape)
print(b.shape)
#L1 SVM
l1svc = LinearSVC(penalty='l1', dual=False).fit(X_1, y_1)

coef = l1svc.coef_.tolist()[0]
#print(coef)
#print(len(l1svc.coef_.tolist()))

print("Number of features having non-zero weight vector coefficients %d " %sum(1 for i in coef if i  != 0))
