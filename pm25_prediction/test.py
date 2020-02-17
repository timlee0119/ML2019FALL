import numpy as np
import pandas as pd
import sys
import os
import csv
from clean import clean_data

def test(testfile, outputfile, best = False):
    features_n = 14 if best else 18 # number of features used

    test_data = pd.read_csv(testfile)

    if best:
        test_data = test_data[test_data["測項"] != "RAINFALL"]
        test_data = test_data[test_data["測項"] != "THC"]
        test_data = test_data[test_data["測項"] != "WD_HR"]
        test_data = test_data[test_data["測項"] != "WIND_DIREC"]


    test_data = clean_data(test_data)
    X_test = []
    for i in range(0, test_data.shape[0], features_n):
        X_test.append(test_data[i:i+features_n,:].ravel())
    X_test = np.array(X_test)
    # print(X_test)
    X_test = np.concatenate((np.ones((X_test.shape[0],1)), X_test), axis = 1)
    weightsfile = "./models/weights-best.npy" if best else "./models/weight.npy"
    Y_test = np.dot(X_test, np.load(weightsfile))

    # create output directory if not exist
    dirname = os.path.dirname(outputfile)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(outputfile, "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["id", "value"])
        for i, y in enumerate(Y_test):
            writer.writerow(["id_%d" % i, y])

if __name__ == "__main__":
    if len(sys.argv) > 3 and sys.argv[3] == "-b":
        test(sys.argv[1], sys.argv[2], True)
    else:
        test(sys.argv[1], sys.argv[2])
