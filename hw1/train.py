import numpy as np
import pandas as pd
import math
import sys
from clean import clean_data

def flatten_data_remove_outliers(data, features_n):
    N = data.shape[0] // features_n
    temp = data[:features_n, :]
    for i in range(1, N):
        temp = np.hstack((temp, data[i*features_n: i*features_n+features_n, :]))

    # replace outliers with mean values
    for i in range(temp.shape[0]):
        t = temp[i, :]
        mean, std = np.mean(t), np.std(t)
        low, up = mean - 2 * std, mean + 2 * std
        temp[i, :] = np.where((low < t) * (t < up), t, mean)

    return temp

def valid(x, y, out):
    if y < 0 or y > out:
        return False
    for xi in x[9]:
        if xi < 0 or xi > out:
            return False
    return True

def parse_train_data(data):
    x = []
    y = []
    pm25_outlier = np.mean(data[9]) + 2 * np.std(data[9])
    l = data.shape[1] - 9
    for i in range(l):
        x_tmp = data[:, i:i+9]
        y_tmp = data[9, i+9]
        if valid(x_tmp, y_tmp, pm25_outlier):
            x.append(x_tmp.reshape(-1,))
            y.append(y_tmp)
    return np.array(x), np.array(y)

def train(X_train, Y_train, b = 10.):
    # add bias
    X_train = np.concatenate((np.full((X_train.shape[0], 1), b), X_train), axis = 1)
    w = np.zeros(X_train.shape[1])
    lr = 0.1
    iteration = 10000
    n = X_train.shape[0]
    grad_squared_sum = np.zeros(X_train.shape[1]) # for ada grad

    for t in range(iteration):
        loss = Y_train - np.dot(X_train, w)
        RMSE = math.sqrt(np.sum(loss ** 2) / n)
        grad = -2 * np.dot(X_train.T, loss)

        grad_squared_sum += grad ** 2
        w = w - lr * grad / np.sqrt(grad_squared_sum)

        print("Iteration %d: RMSE = %f" % (t, RMSE))

    np.save("./models/weights-best", w)
    return w

if __name__ == "__main__":
    features_n = 14 # number of features used
    train_data_list = []
    for i in range(1, len(sys.argv)):
        train_data_list.append(pd.read_csv(sys.argv[i]))
    data = clean_data(pd.concat(train_data_list))
    data = flatten_data_remove_outliers(data, features_n)
    X_train, Y_train = parse_train_data(data)
    w = train(X_train, Y_train)