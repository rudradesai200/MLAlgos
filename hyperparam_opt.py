import numpy as np
from hyperopt import fmin, hp, tpe, Trials
from hyperopt.pyll.base import scope

from functools import partial

from DecisionTreeContinous import DecisionTree
from Random_Forest import RandomForest
from AdaBoost import AdaBoost
from KernelLogisticRegression import KernelLogistRegression

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score


def load_dataset(loc):
    ds = np.load(loc)
    return (ds['arr_0'],ds['arr_1'],ds['arr_2'],ds['arr_3'])

def optimize(result, X, y, Xtest, Ytest):
    print(result)
    model = KernelLogistRegression("rbf",result["eta"],max_iter=result["niters"],reg_param=result["reg_param"],kernel_param=result["kernel_param"])
    # model = RandomForest(result["num_trees"],result["stop"],result["max_level"])

    # kf = StratifiedKFold(n_splits = 3)
    # accuracies = []
    # for idx in kf.split(X=X, y=y):
    #     train_idx, test_idx = idx[0], idx[1]
    #     xtrain, ytrain = X[train_idx], y[train_idx]
    #     xtest, ytest = X[test_idx], y[test_idx]
    #     model.fit(xtrain,ytrain)
    #     preds = model.predict(xtest)
    #     fold_acc = accuracy_score(ytest, preds)
    #     accuracies.append(fold_acc)

    model.fit(X,y)
    preds = model.predict(Xtest)
    # assert(preds.shape == Ytest.shape)
    acc = np.mean(preds.reshape(-1,1) == Ytest.reshape(-1,1))
    print(acc)
    return -1.0 * acc


Xtrain,Ytrain,Xtest,Ytest = load_dataset("Data/dataset_D.npz")


param_space = {
    "reg_param" : hp.choice("reg_param",[0,1e-3]),
    "kernel_param" : scope.int(hp.quniform("kernel_param",3,5,1)),
    "eta" : hp.choice("eta",[1e-5,2e-5,3e-5,4e-5]),
    "niters" : scope.int(hp.quniform("niters",300,700,100)),
    # "num_trees" : scope.int(hp.quniform("num_trees",10,100,10)),
    # "stop" : hp.quniform("stop",10,200,10),
    # "max_level" : hp.quniform("max_level",2,10,1)
}

optimization_func = partial(optimize, X=Xtrain, y=Ytrain, Xtest=Xtest, Ytest=Ytest)

trials = Trials()

result = fmin(
    fn=optimization_func,
    space=param_space,
    algo=tpe.suggest,
    max_evals = 50,
    trials = trials,
)

# result = {'criterion': 1, 'max_level': 3.0, 'stop': 41.0}

print(result)
    # result = {'criterion': 1, 'max_level': 10.0, 'stop': 2.0}
    # model = RandomForest(int(result["num_trees"]),result["stop"],result["max_level"])
    # tree = DecisionTree(["accuracy","entropy"][result["criterion"]],result["stop"],result["max_level"])
    # model.fit(Xtrain,Ytrain)
    # ypred = model.predict(Xtest)
    # print(np.mean(np.array(ypred).reshape(-1,1) == np.array(Ytest).reshape(-1,1)))



## Decision Trees
# Dataset A
# {'criterion': 1, 'max_level': 10.0, 'stop': 2.0} - 0.984
# Dataset B
# {'criterion': 1, 'max_level': 7.0, 'stop': 41.0} - 0.794
# Dataset C
# {'criterion': 1, 'max_level': 9.0, 'stop': 3.0} - 0.8888888888888888
# Dataset D
# {'criterion': 1, 'max_level': 5.0, 'stop': 17.0} - 0.9171597633136095


## Random Forests
# Dataset A
# {'max_level': 8.0, 'num_trees': 90.0, 'stop': 18.0} - 0.99
# Dataset B
# {'max_level': 10.0, 'num_trees': 50.0, 'stop': 77.0} - 0.786
# Dataset C
# {'max_level': 10.0, 'num_trees': 7.0, 'stop': 40.0} - 0.92255
# Dataset D
# {'max_level': 6.0, 'num_trees': 80.0, 'stop': 20.0} - 0.94674