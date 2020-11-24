from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd


def load_dataset(loc):
    ds = np.load(loc)
    X_train = ds['arr_0']
    X_test = ds['arr_2']
    Y_train = ds['arr_1']
    Y_test = ds['arr_3']
    # df = pd.DataFrame()
    # df["x1"] = X_train[:][0]
    # df["x2"] = X_train[:][1]
    # df["y"] = Y_train
    # test_df = pd.DataFrame()
    # test_df["x1"] = X_test[:][0]
    # test_df["x2"] = X_test[:][1]
    # test_df["y"] = Y_test
    return (X_train,Y_train,X_test,Y_test)

if __name__ == "__main__":
    (X_train,Y_train,X_test,Y_test) = load_dataset("../Data/dataset_A.npz")
    # limit = (3 * (len(X_train)))//4
    limit = len(X_train)//4
    incr = (limit)//(40)
    # for level in range(1,6):
    # for stop in range(2,limit,incr):
    for nest in range(100,200,100):
        # for level in range(1,8):
        for stop in range(2, limit, incr):
            forest = RandomForestClassifier(n_estimators=nest,criterion="entropy", min_samples_split=stop,max_features=0.5)
            forest.fit(X_train,Y_train)
            Y_pred = forest.predict(X_test)
            print(f"nest - {nest} , level - inf, stop - {stop} , accuracy - {np.mean(Y_pred == Y_test)}")



################# Decision Trees ########################
### Dataset A -> level - 5, stop - 2, accuracy - 0.994
### Dataset B -> level - 2, stop - 2, accuracy - 0.778
### Dataset C -> level - 2, stop - 2, accuracy - 0.88
### Dataset D -> level - 3, stop - 2, accuract - 0.952

### Dataset A -> level - inf, stop - 2, accuracy - 0.994
### Dataset B -> level - inf, stop - 50, accuracy - 0.79
### Dataset C -> level - inf, stop - 2, accuracy - 0.90
### Dataset D -> level - inf, stop - 45, accuracy - 0.94

################ Random Forest ########################
### Dataset A -> nest - 100 , level - 5, stop - 2 , accuracy - 0.988
### Dataset B -> nest - 100 , level - 4, stop - 2 , accuracy - 0.796
### Dataset C -> nest - 100 , level - 6, stop - 2 , accuracy - 0.959
### Dataset D -> nest - 100 , level - 3, stop - 2 , accuracy - 0.976

### Dataset A -> nest - 100 , level - inf, stop - 20 , accuracy - 0.994
### Dataset B -> nest - 100 , level - inf, stop - 20 , accuracy - 0.798
### Dataset C -> nest - 100 , level - inf, stop - 2 , accuracy - 0.962
### Dataset D -> nest - 300 , level - inf, stop - 2 , accuracy - 0.976



