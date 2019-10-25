import numpy as np
import pandas as pd
import sys
import os

def write_ans(ans, ansfile):
    print("Writing answer to %s" % ansfile)
    import csv
    with open(ansfile, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label"])
        for i in range(1, len(ans)+1):
            writer.writerow([i, ans[i-1]])

def sigmoid(z):
    res = 1. / (1 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)

def train_generative(X_train, Y_train):
    n1, n2 = 0, 0
    mu1, mu2 = np.zeros((X_train.shape[1],)), np.zeros((X_train.shape[1],))
    for i in range(X_train.shape[0]):
        if Y_train[i] == 1:
            n1 += 1
            mu1 += X_train[i]
        else:
            n2 += 1
            mu2 += X_train[i]
    mu1 /= n1
    mu2 /= n2
    
    sig1 = np.zeros((X_train.shape[1], X_train.shape[1]))
    sig2 = np.zeros((X_train.shape[1], X_train.shape[1]))
    for i in range(X_train.shape[0]):
        if Y_train[i] == 1:
            sig1 += np.dot(np.transpose([X_train[i] - mu1]), [(X_train[i] - mu1)])
        else:
            sig2 += np.dot(np.transpose([X_train[i] - mu2]), [(X_train[i] - mu2)])
    sig1 /= n1
    sig2 /= n2
    sig = (n1 / X_train.shape[0]) * sig1 + (n2 / X_train.shape[0]) * sig2
    
    return mu1, mu2, sig, n1, n2

def predict_generative(X_test, mu1, mu2, sig, n1, n2):
    inv_sig = np.linalg.inv(sig)
    w = np.dot((mu1 - mu2), inv_sig)
    b = (-0.5) * np.dot(np.dot(mu1.T, inv_sig), mu1) + (0.5) * np.dot(np.dot(mu2.T, inv_sig), mu2) + np.log(float(n1)/n2)
    z = np.dot(w, X_test.T) + b
    pred = sigmoid(z)
    return pred

if __name__ == "__main__":
    hlp = "python3 generative.py train.csv test.csv X_train Y_train X_test ans.csv"
    if len(sys.argv) != 7:
        print(hlp)
        sys.exit(0    # y_pred = predict_generative(X_train, mu1, mu2, sig, n1, n2)
    # y_pred = np.around(y_pred)
    # result = (Y_train.ravel() == y_pred)
    # print('Train acc = %f' % (float(result.sum()) / result.shape[0]))
)

    X_train_raw = pd.read_csv(sys.argv[3])
    Y_train_raw = pd.read_csv(sys.argv[4])
    X_test_raw = pd.read_csv(sys.argv[5])

    X_train = X_train_raw.values.astype(float)
    X_test = X_test_raw.values.astype(float)
    Y_train = np.concatenate(([[0]], Y_train_raw.values))

    # remove native_country
    X_train = X_train[:, :64]
    X_test = X_test[:, :64]

    mu1, mu2, sig, n1, n2 = train_generative(X_train, Y_train)
    y_pred = predict_generative(X_test, mu1, mu2, sig, n1, n2)
    y_pred = np.around(y_pred).astype(np.int)
    write_ans(y_pred, sys.argv[6])