import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.shape_base import split
import pandas as pd
import math

NDIVS = 100

def find_entropy(df,selector,target,split_type):
    """Helper function for calculate_entropy. Calculates entropy for the 
    given target with given selector

    Args:
        df (pandas.DataFrame): Dataset
        selector (boolean array): selects the part of dataset
        target (string): the target category

    Returns:
        float: Final Entropy
    """
    s = set(df[target][selector])
    entropy = 0.0
    for cat in s:
        probab = np.mean(df[target][selector] == cat)
        # print(probab)
        entropy += (-probab)*(math.log2(probab))
    return entropy

def calculate_entropy(df,attr_list, split_type, target):
        """Calculates entropy for all the attr in attr_list for the given dataframe.

        Args:
            df (pandas.DataFrame): Dataset having all the attr in attr_list
            attr_list (list): List of attributes for finding entropy

        Returns:
            (attr,entropy): Attr having min entorpy
        """
        if split_type == "categorical":
            entropy_list = []
            for attr in attr_list:
                final_entropy = 0.0
                cats = set(df[attr])
                for cat in cats:
                    probab = np.mean(df[attr] == cat)
                    final_entropy += probab*find_entropy(df,(df[attr]==cat),target)
                entropy_list.append(final_entropy)
            if(len(entropy_list) == 1):
                return attr_list[0],entropy_list[0]
            else:
                return (attr_list[np.argmin(entropy_list)],np.min(entropy_list))
        elif split_type == "continous":
            ### Left to be done
            entropy_list = []
            for attr in attr_list:
                final_entropy = 0.0
                min_v,max_v = min(df[attr]), max(df[attr])
                increment = (max_v-min_v)//NDIVS
                for i in range(min_v,max_v+1,increment):
                    probab = np.mean(df[attr] < i)
                entropy_list.append(final_entropy)

            if(len(entropy_list) == 1):
                return attr_list[0],entropy_list[0]
            else:
                return (attr_list[np.argmin(entropy_list)],np.min(entropy_list))

def calculate_accuracy(df,attr_list, split_type,target):
    return 0

def create_tree(df,attr_list,target,currlevel,maxlevel,calculate_criterion,split_type):
    """Helper function. Generates tree for the given df,attr_list and target.

    Args:
        df (pandas.DataFrame): DataFrame for the given dataset
        attr_list (string_list): list of splittable attributes 
        target (string): target attribute

    Returns:
        Decision_Tree_Node: Node representing the given dataframe
    """
    if(len(attr_list) == 0):
        return None
    elif((len(attr_list) == 1) or (currlevel == maxlevel)):
        attr,min_criterion_value,split_value = calculate_criterion(df,attr_list,target)     # Calculates the min value of the criterion
        res = max(set(df[target]), key = list(df[target]).count)            # Makes decision as the one with max probab
        return Decision_Tree_Node(attr,min_criterion_value,0,split_type,res)    # Create leaf node
    else:
        attr,min_criterion_value,split_value = calculate_criterion(df,attr_list,split_type,target)
        if(len(set(df[target])) == 1):
            return Decision_Tree_Node(attr,min_criterion_value,0,split_type,list(set(df[target]))[0]) # Only one class remaining
        else:
            rem_attr = attr_list.copy()
            rem_attr.remove(attr)
            if split_type == "categorical":
                vals = set(df[attr])
                node = Decision_Tree_Node(attr,min_criterion_value,len(vals),split_type)
                for val in vals:
                    node.add_branch(val,split_type,create_tree(df[df[attr] == val],rem_attr,target,currlevel+1,maxlevel,split_type))
                return node
            elif split_type == "continuos":
                node = Decision_Tree_Node(attr,min_criterion_value,2,split_type,None,split_value)
                node.add_branch("left",split_type,create_tree(df[df[attr] < split_value],rem_attr,target,currlevel+1,maxlevel,calculate_criterion,split_type))
                node.add_branch("right",split_type,create_tree(df[df[attr] >= split_value],rem_attr,target,currlevel+1,maxlevel,calculate_criterion,split_type))
                return node
                

class Decision_Tree_Node:
    """Class for Decision Tree Node
    Attributes:
    split_attr  {string}    -> Attribute which splitted this node
    entropy     {float}     -> Calculated entropy for the given split
    n_childs    {int}       -> Number of childs of this node
    is_leaf     {bool}      -> True if this node is a leaf
    decision    {target}    -> Prediction for leaf 
    """


    def __init__(self,split_attr,criterion_val,n_childs,split_type,decision=None):
        self.split_attr = split_attr
        self.criterion_val = criterion_val
        self.n_childs = n_childs
        self.split_type = split_type
        self.branches = dict()
        self.decision = decision
        self.is_leaf = (n_childs == 0)
    

    def add_branch(self,val,split_type,node):
        """Adds branch according to the val of split_attr

        Args:
            val (string): Value of the split_attr
            node (Decision_Tree_Node): Corresponding node
        """
        self.branches[val] = node
    
    def predict(self,obs):
        """Predicts the final class for given observation

        Args:
            obs (dict): Observation with all the required attributes
            Can be dataframe row

        Returns:
            prediction: Final prediction. Returns None, if no such splits
        """
        if(self.is_leaf):
            return self.decision
        else:
            try:
                nd = self.branches[obs[self.split_attr]]
            except:
                return None
            if(nd != None):
                return nd.predict(obs)
            else:
               return  None
            
    def print(self,prefix="",isLast=True):
        """Pretty prints the tree

        Args:
            prefix (str, optional): The prefix for printing. Defaults to "".
            isLast (bool, optional): Boolean to check if this is the last node. Defaults to True.
        """
        if(self == None):
            return
        prstr = prefix
        if(isLast):
            prstr += "└─"
        else:
            prstr += "├─"
        if(self.is_leaf):
                print(prstr,self.split_attr,"--->",self.decision)
        else:
            print(prstr,self.split_attr)
        for i,(b,v) in enumerate(self.branches.items()):
            if(v != None):
                if(i == (self.n_childs - 1)):
                    v.print(prefix + f"| {b} ",True)
                else:
                    v.print(prefix+f"| {b} ",False)
                
class Decision_Tree:
    """Decision Tree Class. Uses Entropy as the split param. Follows ID3 algorithm
    
    Methods:
        calculate_entropy(df,attr_list) : Returns attribute with the min entropy
        create_tree(df,attr_list,target) : Helper Function. Generates a tree from the given dataframe
        print_stats(): Prints Stats for the given dataframe and tree
        generate_tree() : Creates a Decision Tree for the given Dataframe
    """
    def __init__(self,df,target,type="categorical",criterion="entropy",level=-1):
        self.df = df
        self.target = target
        self.attr_list = list(df.columns)
        self.attr_list.remove(target)
        self.root = None
        self.criterion = criterion
        self.type = type
        if(level == -1):
            self.maxlevel = len(self.attr_list) + 1
        else:
            self.maxlevel = level
    
    def print_stats(self,testdf):
        """prints stats like accuracy for the tree
        """
        acc = 0.0
        for i in range(len(testdf)):
            acc += (testdf.iloc[i][self.target] == self.root.predict(testdf.iloc[i]))
        acc /= (len(testdf))
        print("Accuracy",acc)
        return acc
        
    def generate_tree(self):
        """Generates Tree with the data initialized. Also prints the stat with it.
        """
        calculate_criterion = None
        if(self.criterion == "entropy"):
            calculate_criterion = calculate_entropy
        elif self.criterion == "accuracy":
            calculate_criterion = calculate_accuracy
        self.root = create_tree(self.df,self.attr_list,self.target,1,self.maxlevel,calculate_criterion)
        return self.print_stats(self.df)
    
    def generate_preds(self,testdf):
        """Generates predictions for the test dataframe

        Args:
            testdf (pandas.Dataframe): Test dataframe

        Returns:
            DataFrame: Test DataFrame with predictions stored in target
        """
        retdf = testdf.copy()
        for i in range(len(testdf)):
            retdf[self.target] = self.root.predict(testdf.iloc[i])
        return retdf