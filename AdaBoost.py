from operator import le
from random import sample
import numpy as np
import pandas as pd
from DecisionTreeContinous import DecisionTree
import matplotlib.pyplot as plt
import copy

class AdaBoost:
    """
    Contains AdaBoost algorithm implementation.
    Accepts any weaklearner with fit(X,y) and predict(X) methods.
    """
    def __init__(self,n_estimators=10,verbose=False,nstop=2,maxl=2):
        """Initializes AdaBoost Classifier

        Args:
            weak_learner (Object): Weaklearner to train Adaboost.
                Should Contain fit(X,y) and predict(X) method.
                And __copy() which returns a new weaklearner with same params
            n_estimators (int, optional): Number of weaklearners to train. Defaults to 10.
            verbose (bool, optional): Get some output. Defaults to False.
        """
        # self.__weak_learner = DecisionTree("entropy",max_level=maxl,num_nodes_stop=nstop)
        self.nstop = nstop
        self.maxl = maxl
        self.__n_estimators = n_estimators
        self.__sample_weights = []
        self.__stumps = []
        self.__stump_weights = []
        self.__train_accs = np.zeros(self.__n_estimators,dtype=np.float)
        self.__verbose = verbose

    def get_train_accs(self): return self.__train_accs  # Used to get train_accs after training

    def predict(self, X):
        """
        Predict class for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray of shape (n_samples,) or (n_samples, n_outputs)
            The predicted classes.
        """
        stump_preds = np.array([(stump.predict(X)) for stump in self.__stumps])
        return np.sign(np.dot(self.__stump_weights, stump_preds))

    def get_test_accs(self,X,y):
        """
        Get Test Accuracies for the dataset (X,y)
        """
        test_accs = []
        preds = np.zeros(len(X),dtype=np.float)
        for i in range(self.__n_estimators):
            pred = self.__stumps[i].predict(X)
            preds += (self.__stump_weights[i] * pred)
            test_accs.append(np.mean(np.sign(preds/(i+1)) == y))
        test_accs = np.array(test_accs)
        return test_accs

    def fit(self, X, y):
        """
        Builds a Ada Boost classifier for the training dataset (X,y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, n_outputs)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
        """
        n = len(X)
        # init numpy arrays
        stumps = np.zeros(shape=self.__n_estimators,dtype=object)
        # for t in range(self.__n_estimators):
        #     stumps[t] = weaklearner._copy()
        
        stump_weights = np.zeros(shape=self.__n_estimators)
        sample_weights = np.zeros(shape=(self.__n_estimators,n))
        preds = np.zeros(shape=n,dtype=np.float)
        train_accs = np.zeros(self.__n_estimators)
        # initialize weights uniformly
        sample_weights[0] = (np.ones(shape=n) / n)

        for t in range(self.__n_estimators):
            
            # fit  weak learner
            curr_sample_weights = sample_weights[t]
            stump = DecisionTree("entropy",self.nstop,self.maxl)
            stump.fit(X, y, curr_sample_weights)

            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            err = curr_sample_weights[(stump_pred != y)].sum()# / n        
            stump_weight = np.log((1 - err) / err) / 2

            # update sample weights
            new_sample_weights = curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
            new_sample_weights /= new_sample_weights.sum()

            # If not final iteration, update sample weights for t+1
            if t+1 < self.__n_estimators: sample_weights[t+1] = (new_sample_weights)

            # save results of iteration
            stumps[t] = (stump)
            stump_weights[t] = (stump_weight)
            preds += (stump_weight * stump_pred)
            train_accs[t] = np.mean(np.sign(preds/(t+1)) == y)
            if self.__verbose: print(f"Iteration {t}: accuracy = {train_accs[t]}")

        self.__sample_weights = sample_weights
        self.__stump_weights = stump_weights
        self.__stumps = stumps
        self.__train_accs = train_accs
        return self

def train_ada_boost(X, y, num_nodes_stop=2, level=3, iters=10):
    """ Returns a random forest trained on X and Y. 
    Trains num_trees.
    Stops splitting nodes in each tree when a node has hit a size of "num_nodes_stop" or lower.
    Split criterion can be either 'accuracy' or 'entropy'.
    Fraction of data used per tree = a
    Fraction of features used in each node = b
    Returns a random forest (In whatever format that you find appropriate)
    """
    # weaklearner = DecisionTree("accuracy",num_nodes_stop,level)
    adaboostclf = AdaBoost(iters,True,nstop=num_nodes_stop,maxl=level)
    adaboostclf.fit(X,y)
    return adaboostclf


def eval_ada_boost(adaboostclf, test_X):
    """ Takes in a  random forest object (hhowever you want to store it), and a bunch of instances X and 
    returns the tree predicted values at those instances."""
    return adaboostclf.predict(test_X)

if __name__ == "__main__":
    def load_dataset(loc):
        ds = np.load(loc)
        X_train = ds['arr_0']
        X_test = ds['arr_2']
        Y_train = ds['arr_1']
        Y_test = ds['arr_3']
        return (X_train,Y_train,X_test,Y_test)

    def plot(adaboostclf, DS="A"):
        (X_train,Y_train,_,_) = load_dataset(f"../Data/dataset_{DS}.npz")
        fig,axs = plt.subplots(figsize=(10,10))
        n1,n2 = 100,100
        if DS == "A":
            l12,l22,l11,l21 = -1,1.5,-1.5,2.5
        else:
            l11,l21,l12,l22 = -1.3,1.3,-1.3,1.3
        x1,x2 = np.linspace(l11,l21,n1),np.linspace(l12,l22,n2)
        x1,x2 = np.meshgrid(x1,x2)
        x1,x2 = x1.reshape((n1*n2),1),x2.reshape((n1*n2),1)
        X = np.concatenate((x1,x2),axis=1)
        X1,X2 = x1.reshape((n1,n2)),x2.reshape((n1,n2))
        Y = np.array(eval_ada_boost(adaboostclf,X)).reshape(n1,n2)
        ax = axs
        a = ax.contourf(X1,X2,Y,colors = ['#ffcccc','#99ffbb'])
        proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in a.collections]
        contour_legend = ax.legend(proxy[::-1],["classified as Y = 1","classified as Y = -1"])
        ax.add_artist(contour_legend)
        ax.set_ylabel(f"Dataset {DS}")
        ax.set_title("Num_nodes_stop = 2, criterion = accuracy")
        X_train_1 = X_train[Y_train == 1]
        X_train_2 = X_train[Y_train == -1]
        ax.scatter(X_train_1[:,0],X_train_1[:,1],marker="+",c='#339966',label='Y_train = 1')    
        ax.scatter(X_train_2[:,0],X_train_2[:,1],marker="x",c='#ff1a1a',label='Y_train = -1')
        ax.legend(loc="lower left")
        plt.show()

    def plotline(n,train,test):
        ns = np.array([i for i in range(1,n+1)])
        plt.plot(ns, train, color='g', label="Train_acc")
        plt.plot(ns, test, color='orange', label="Test_acc")
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc="lower left")
        plt.title('Accuracy vs Epochs')
        plt.show()

    ds = "C"
    iter = 10
    level = 5
    X_train,Y_train,X_test,Y_test = load_dataset(f"../Data/dataset_{ds}.npz")
    adaboostclf = train_ada_boost(X_train,Y_train,level=level,iters=iter,num_nodes_stop=100)
    pred = eval_ada_boost(adaboostclf,X_test)
    train_accs = adaboostclf.get_train_accs()
    test_accs = adaboostclf.get_test_accs(X_test,Y_test)

    plotline(iter,train_accs,test_accs)
    print(f"Final Accuracy = {np.mean(pred == Y_test)}")
    plot(adaboostclf,ds)