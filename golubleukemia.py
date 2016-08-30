import numpy as np
import csv
from sklearn.datasets import load_svmlight_file




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
    print(len(score))
    return (score, score)  # return score as score , score as asked


def main():
    data = load_svmlight_file("leumerged")
    X_1 = data[0].todense().tolist()
    y_1 = map(int,data[1])

    lables= []
    for i in range(0,len(y_1)):
        lables = lables + [[y_1[i]]]

    #print(X_1)
    #print(y_1)
    #print(lables)
    print golub(X_1,lables)

if __name__ == "__main__":
    main()
