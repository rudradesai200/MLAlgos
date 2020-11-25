import numpy as np
import pandas as pd
import math
from random import randrange
import matplotlib.pyplot as plt


class DecisionTree:
    """
    Decision Tree Class.
    Criterions implemented = ["entropy","accuracy"]
    Regularizations implemented = num_nodes_stop, max_level
    """
    
    ## Some Constants
    NDIV = 10
    MAX_INT = np.iinfo(np.int32).max

    class __DecisionTreeNode:
        """
        Base Class for Decision Tree.
        Not to be used directly.
        """
        def __init__(self,attr,val,is_leaf=False,decision=None):
            self.__split_attr = attr
            self.__split_val = val
            self.__is_leaf = is_leaf
            self.__branches = dict()
            self.__decision = decision

        def predict(self,obs):
            """Predicts label for the given obs.
            Expects obs to be a pandas row.
            """
            if(self.__is_leaf): return self.__decision
            else:
                nd = self.__branches["left"] if(obs[self.__split_attr] <= self.__split_val) else self.__branches["right"]
                if(nd != None): return nd.predict(obs)
                else: raise ValueError("Node == None. Error building the tree.")

        def add_branch(self,val,nd):
            self.__branches[val] = nd

        def print(self,prefix="",isLast=True):
            """Pretty prints the Node"""
            if(self == None): return
            prstr = prefix + ("└─" if isLast else "├─")

            if(self.__is_leaf): print(prstr,self.__decision)
            else: print(prstr,self.__split_attr,self.__split_val)

            for i,(b,v) in enumerate(self.__branches.items()):
                if(v != None):
                    if(i == (1)): v.print(prefix + f"| {b} ",True)
                    else: v.print(prefix+f"| {b} ",False)

    def __init__(self,criterion="entropy",num_nodes_stop=2,max_level=-1,feature_sampling=False,b=None):
        """Initializes a DecisionTree Object.

        Args:
            criterion (str, optional): Critierion to split. Options - \"entropy\", \"accuracy\". Defaults to "entropy".
            num_nodes_stop (int, optional): Limit to stop splitting if num_obs < num_nodes_stop. Defaults to 2.
            max_level (int, optional): Limit to stop splitting if num_obs . Defaults to MAX_INT.
            feature_sampling (bool, optional): Turn on feature sampling. Defaults to False.
            b (float, optional): Fraction of features to sample. Defaults to None.
        """
        self.root = None
        self.criterion = criterion
        self.num_nodes_stop = num_nodes_stop
        if max_level > 0:
            self.max_level = max_level
        else:
            self.max_level = self.MAX_INT
        self.b = b
        self.feature_sampling = feature_sampling
    
    def __convert_np_to_df(X,y=[]):
        (_,d) = X.shape
        df = pd.DataFrame()
        for i in range(d): df[f"x{i+1}"] = X[:,i]
        target = None
        if len(y):
            target = "y"
            df[target] = y
        return df,target
    
    def fit(self, X, y, weights=[]):
        """
        Builds a Decision Tree for the Traning Dataset (X,y).

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The training input samples.

        y : array-like of shape (n_samples,) or (n_samples, 1)
            The target values (class labels in classification, real numbers in
            regression).

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted. Splits
            with zero elements on either sides are ignored.

        Returns
        -------
        self : object
        """
        self.df,self.target = DecisionTree.__convert_np_to_df(X,y)
        if len(weights) == 0:
            self.df["weights"] = np.ones(len(self.df),dtype=np.float)
        else:
            self.df["weights"] = weights
        self.attrs = list(self.df.columns)
        self.attrs.remove(self.target)
        self.attrs.remove("weights")
        self.root = self.__create_node(self.df,1)
        return self

    def __entropy(pr): return 0 if((pr <= 0) or (pr >= 1)) else -((pr * (math.log2(pr))) + (1-pr)*(math.log2(1-pr)))

    def __entropy_of_split(self,df,attr):
        """Private Function. Finds the minimum entropy of split for continous attributes."""
        s = list(set(df[attr]))
        start,end = 0,len(s)
        incr = 1 if(len(s) < DecisionTree.NDIV) else (len(s)//DecisionTree.NDIV)
        min_entropy = 1e100
        split_value = None
        if incr != 0:

            for j in range(start,end,incr):
                # Calculate Probabilities for passing to entropy.
                pl = round(np.dot((df[attr] <= s[j]),df["weights"])/np.sum(df["weights"]),4)
                pr = round(1 - pl,4)
                tl,tr = df["weights"][df[attr] <= s[j]],df["weights"][df[attr] > s[j]] 
                ql = 0 if(pl <= 0) else round(np.dot((df[self.target][df[attr] <= s[j]] == 1),tl)/np.sum(tl) ,4)
                qr = 0 if(pr <= 0) else round(np.dot((df[self.target][df[attr] > s[j]] == 1),tr)/np.sum(tr),4)
                curr_entropy = pl*DecisionTree.__entropy(ql) + pr*DecisionTree.__entropy(qr)
                # Find the min entropy and split_value
                min_entropy = min(min_entropy, curr_entropy)
                if(min_entropy == curr_entropy): split_value = s[j]
        
        return min_entropy,split_value

    def __accuracy_of_split(self,df,attr):
        """ Private Function. Finds the minimum accuracy of split for continous attributes.
            Here, we are choosing the min value as max(acc) = min(1-acc).
            This helps to keep a common interface with entropy.
        """
        s = list(set(df[attr]))
        start,end = 0,len(s)
        incr = 1 if(len(s) < DecisionTree.NDIV) else (len(s)//DecisionTree.NDIV)
        min_acc,split_value = 1e100, None
        if incr != 0:
            for j in range(start,end,incr):
                # Calculate Probabilities
                wsum = np.sum(df["weights"])
                a = np.dot((df[self.target][df[attr] <= s[j]] == 1),df["weights"][df[attr] <= s[j]])
                b = np.dot((df[attr] <= s[j]),df["weights"])
                x = np.dot((df[self.target] == 1),df["weights"]) - a
                y = wsum - b
                # Zero Split
                if((b==0)or(y==0)): continue
                curr_acc = (a + y - x)/(b+y)
                min_curr_acc = min(curr_acc,1-curr_acc)
                min_acc = min(min_acc, min_curr_acc)
                if(min_acc == min_curr_acc): split_value = s[j]
        return min_acc,split_value

    def __calculate_criterion(self,df,attr_list):
        """Private Function. Currently entropy and accuracy supported
        Calculates the min_criteria, split_attr, and split_value for continous attributes.
        """
        if(self.criterion == "entropy"): criterion_val = self.__entropy_of_split
        elif(self.criterion == "accuracy"): criterion_val = self.__accuracy_of_split
        else: raise ValueError(f"Unknown Criteria Passed - {self.criterion}")

        ## find the min criteria by looping over all the attributes.
        min_criterion, split_attr, split_value = 1e100, None, None
        for attr in attr_list:
            temp_criterion,temp_split = criterion_val(df,attr)
            if(temp_criterion < min_criterion):
                min_criterion = temp_criterion
                split_attr = attr
                split_value = temp_split
            
        if split_value == None: raise ValueError("No valid splits found")
        return split_attr,split_value

    def __sample_features(self,attrs):
        """
        Private Function.
        Used only in case of Random Forest for sampling features.
        """
        features = list()
        while len(features) < round(self.b * len(attrs)):
            index = randrange(len(attrs))
            if attrs[index] not in features:
                features.append(attrs[index])
        return features

    def __create_node(self,df,currlevel):
        """
        Private Function.
        Used for creating DecisionTree Nodes.
        """
        # print(len(df),currlevel)
        if(len(df) == 0): return None # No training Input. Gives Error in predict stage
        elif((len(df) <= self.num_nodes_stop) or (currlevel >= self.max_level) or (len(set(df[self.target])) == 1)):
            # OverFitting Criterias reached or Only one majority is left
            res = max(list(df[self.target]), key = list(df[self.target]).count)
            return DecisionTree.__DecisionTreeNode(None,None,True,res)
        else:
            attrs = self.__sample_features(self.attrs) if self.feature_sampling else self.attrs
            attr,split_val = self.__calculate_criterion(df,attrs)
            if((len(df[df[attr] <= split_val]) == 0) or (len(df[df[attr] > split_val]) == 0)):
                raise ValueError("Zero split detected. Left or Right split empty.")
            else:
                node = DecisionTree.__DecisionTreeNode(attr,split_val)
                node.add_branch("left",self.__create_node(df[df[attr] <= split_val],currlevel+1))
                node.add_branch("right",self.__create_node(df[df[attr] > split_val],currlevel+1))
                return node

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
        y = np.zeros(shape=len(X),dtype=np.float)
        test_df,_ = DecisionTree.__convert_np_to_df(X)
        for i in range(len(X)): y[i] = self.root.predict(test_df.iloc[i])
        return y
    
def train_decision_tree(X, y, num_nodes_stop=2, level=-1, criterion='accuracy'):
    """ Returns a decision tree trained on X and Y. 
    Stops splitting nodes when a node has hit a size of "num_nodes_stop" or lower.
    Split criterion can be either 'accuracy' or 'entropy'.
    Returns a tree (In whatever format that you find appropriate)
    """
    tree = DecisionTree(criterion,num_nodes_stop,level)
    tree.fit(X,y)
    return tree

def eval_decision_tree(tree, test_X):
    """ Takes in a tree, and a bunch of instances X and 
    returns the tree predicted values at those instances."""
    return tree.predict(test_X)