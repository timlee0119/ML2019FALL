import numpy as np
import pandas as pd
import sys
import os

def normalize(X_train, X_test):
    # normalize numerical data
    X_all = np.concatenate((X_train, X_test), axis = 0)
    X_train_norm = X_train.copy()

    numerical_col_idx = [0, 1, 3, 4, 5]
    for col in numerical_col_idx:
        mean, std = np.mean(X_all[:, col]), np.std(X_all[:, col])
        X_train_norm[:, col] = (X_train_norm[:, col] - mean) / std

    return X_train_norm

def sigmoid(z):
    res = 1. / (1 + np.exp(-z))
    return np.clip(res, 1e-6, 1-1e-6)

def train_logistic(X_train, Y_train, X_test, iteration, verbose = True, bias = 1):
    # normalization
    X_train = normalize(X_train, X_test)
    X_train = np.concatenate((np.full((X_train.shape[0], 1), bias), X_train), axis = 1)
    
    w = np.zeros((X_train.shape[1], 1))
    lr = 1e-1
    n = X_train.shape[0]
    grad_squared_sum = np.zeros((X_train.shape[1], 1))

    diff = Y_train - sigmoid(np.dot(X_train, w))
    
    for t in range(iteration):
        y_predict = sigmoid(np.dot(X_train, w))
        grad = -np.dot(X_train.T, Y_train - y_predict)
        grad_squared_sum += grad ** 2
        w = w - lr * grad / np.sqrt(grad_squared_sum)

        if verbose and (t+1) % 500 == 0:
            loss = -1*np.mean(Y_train*np.log(y_predict+1e-100) + (1-Y_train)*np.log(1-y_predict+1e-100))
            print("Iteration %d, diff = %f" % (t, loss))
    return w

def test_logistic(X_train, X_test, w):
    X_test = normalize(X_test, X_train)
    X_test = np.concatenate((np.ones((X_test.shape[0], 1)), X_test), axis = 1)
    Y_test = sigmoid(np.dot(X_test, w))
    ans = (Y_test > 0.5).astype(int).ravel()
    return ans

def write_ans(ans, ansfile):
    print("Writing answer to %s" % ansfile)
    # create output directory if not exist
    dirname = os.path.dirname(ansfile)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    import csv
    with open(ansfile, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "label"])
        for i in range(1, len(ans)+1):
            writer.writerow([i, ans[i-1]])

def calc_acc(y_valid, ans):
    acc = 0
    y_valid = y_valid[:, 0]   # y_valid.shape = (n, 1), ans.shape = (n,)
    for i in range(ans.shape[0]):
        if y_valid[i] == ans[i]:
            acc += 1
    acc /= ans.shape[0]
    return acc

if __name__ == "__main__":
    train = False
    hlp = "python3 logistic.py train.csv test.csv X_train Y_train X_test ans.csv [-t]"
    if len(sys.argv) != 7 and len(sys.argv) != 8:
        print(hlp)
        sys.exit(0)
    if len(sys.argv) == 8 and sys.argv[7] == "-t":
        train = True

    X_train_raw = pd.read_csv(sys.argv[3])
    Y_train_raw = pd.read_csv(sys.argv[4])
    X_test_raw = pd.read_csv(sys.argv[5])

    X_train = X_train_raw.values.astype(float)
    X_test = X_test_raw.values.astype(float)
    Y_train = np.concatenate(([[0]], Y_train_raw.values))

    # remove native_country
    X_train = X_train[:, :64]
    X_test = X_test[:, :64]

    if train:
        print("Start training logistic regression model...")
        w = train_logistic(X_train, Y_train, X_test, 15000)
        np.save("model/logistic-weights", w)
    else:
        w = np.load("model/logistic-weights.npy")
        ans = test_logistic(X_train, X_test, w)
        write_ans(ans, sys.argv[6])
