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

def GBC(X_train, Y_train, X_test):
    from sklearn.ensemble import GradientBoostingClassifier
    clf = GradientBoostingClassifier(n_estimators=250, learning_rate=0.1, random_state=42, min_samples_split=200, min_samples_leaf=50, max_depth=8, max_features='sqrt', subsample=0.8).fit(X_train, Y_train.ravel())
    return clf.predict(X_test)

if __name__ == "__main__":
    hlp = "python3 best.py train.csv test.csv X_train Y_train X_test ans.csv"
    if len(sys.argv) != 7:
        print(hlp)
        sys.exit(0)

    X_train_raw = pd.read_csv(sys.argv[3])
    Y_train_raw = pd.read_csv(sys.argv[4])
    X_test_raw = pd.read_csv(sys.argv[5])

    X_train = X_train_raw.values.astype(float)
    X_test = X_test_raw.values.astype(float)
    Y_train = np.concatenate(([[0]], Y_train_raw.values))

    # remove native_country
    X_train = X_train[:, :64]
    X_test = X_test[:, :64]

    # Gradient Tree Boosting Classifier
    ans = GBC(X_train, Y_train, X_test)
    write_ans(ans, sys.argv[6])
