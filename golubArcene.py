import numpy as np
import csv




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
    print("The number of values in the golub score vector %d" %len(score))
    return (score, score)  # return score as score , score as asked


def main():
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
            b.append(row)   # labels matrix
        print(golub(a,b))
    finally:
        f.close   # close the files
        f1.close


if __name__ == "__main__":
    main()
