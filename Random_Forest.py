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

def train_random_forest(X, y, num_trees=10, num_nodes_stop=2, criterion='accuracy', a=0.5, b=0.5, level=-1,):
    """ Returns a random forest trained on X and Y. 
    Trains num_trees.
    Stops splitting nodes in each tree when a node has hit a size of "num_nodes_stop" or lower.
    Split criterion can be either 'accuracy' or 'entropy'.
    Fraction of data used per tree = a
    Fraction of features used in each node = b
    Returns a random forest (In whatever format that you find appropriate)
    """
    forest = RandomForest(num_trees,num_nodes_stop,level,criterion, a, b)
    forest.fit(X,y)
    return forest
    
def eval_random_forest(random_forest, test_X):
    """ Takes in a  random forest object (hhowever you want to store it), and a bunch of instances X and 
    returns the tree predicted values at those instances."""
    return random_forest.predict(test_X)

if __name__ == "__main__":
    def load_dataset(loc):
        ds = np.load(loc)
        X_train = ds['arr_0']
        X_test = ds['arr_2']
        Y_train = ds['arr_1']
        Y_test = ds['arr_3']
        return (X_train,Y_train,X_test,Y_test)

    def plot(forest, DS="A"):
        (X_train,Y_train,X_test,Y_test) = load_dataset(f"../Data/dataset_{DS}.npz")
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

        Y = np.array(eval_random_forest(forest,X)).reshape(n1,n2)
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

    ds = "A"
    X_train,Y_train,X_test,Y_test = load_dataset(f"../Data/dataset_{ds}.npz")
    forest = train_random_forest(X_train,Y_train,10,20,"entropy",0.5,0.5)
    preds = eval_random_forest(forest,X_test)
    print(np.mean(preds == Y_test))
    plot(forest,ds)