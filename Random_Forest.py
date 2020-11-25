import numpy as np
from DecisionTreeContinous import DecisionTree
import matplotlib.pyplot as plt

class RandomForest:
    """
    RandomForest Implementation with Base Class - DecisionTree
    TODO:
        - Implement Categorical attributes
        - Allow Weighted Samples
    """
    def __init__(self,num_trees=10,num_nodes_stop=2,max_level=-1,criterion="entropy",a=0.5,b=0.5, verbose=False):
        """Initiates a Random Forest

        Args:
            num_trees (int, optional): No of trees in the forest. Defaults to 10.
            num_nodes_stop (int, optional): Regularization for decision tree. Defaults to 2.
            max_level (int, optional): Regularization for Decision Tree. Defaults to -1.
            criterion (str, optional): ["entropy","accuracy"]. Defaults to "entropy".
            a (float, optional): Sample Fraction for individual trees . Defaults to 0.5.
            b (float, optional): Fraction for feature sampling. Defaults to 0.5.
        """
        self.num_trees = num_trees
        self.num_nodes_stop = num_nodes_stop
        self.max_level = max_level
        self.verbose = verbose
        self.criterion = criterion
        self.a = a
        self.b = b
        self.forest = []

    def fit(self,X, y):
        """
        Builds a Random Forest for the training dataset (X,y).

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
        n_sample = round(len(X) * self.a)
        for i in range(self.num_trees):
            if self.verbose: print(f"Training Tree {i+1}")
            sample = np.random.choice(len(X), n_sample, replace=True)
            tree = DecisionTree(self.criterion,self.num_nodes_stop,self.max_level,True,self.b)
            tree.fit(X[sample],y[sample])
            self.forest.append(tree)
        return self

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
        y = np.array([tree.predict(X) for tree in self.forest])
        return np.sign(np.sum(y, axis=0))