import numpy as np
import copy

class AdaBoost:
    """
    Contains AdaBoost algorithm implementation.
    Accepts any weaklearner with fit(X,y) and predict(X) methods.
    """
    def __init__(self,weaklearner,n_estimators=10,verbose=False):
        """Initializes AdaBoost Classifier

        Args:
            weak_learner (Object): Weaklearner to train Adaboost.
                Should Contain fit(X,y) and predict(X) method.
            n_estimators (int, optional): Number of weaklearners to train. Defaults to 10.
            verbose (bool, optional): Get some output. Defaults to False.
        """
        self.__weak_learner = weaklearner
        self.__n_estimators = n_estimators
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
        y : ndarray of shape (n_samples, 1)
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

        y : array-like of shape (n_samples,) or (n_samples, 1)
            The target values (class labels in classification, real numbers in
            regression).

        Returns
        -------
        self : object
        """
        n = len(X)

        # init numpy arrays
        stumps = np.zeros(shape=self.__n_estimators,dtype=object)
        stump_weights = np.zeros(shape=self.__n_estimators)
        sample_weights = np.zeros(shape=(self.__n_estimators,n))
        preds = np.zeros(shape=n,dtype=np.float)
        train_accs = np.zeros(self.__n_estimators)
        sample_weights[0] = (np.ones(shape=n) / n)

        for t in range(self.__n_estimators):
            
            # fit  weak learner
            curr_sample_weights = sample_weights[t]
            stump = copy.deepcopy(self.__weak_learner)
            stump.fit(X, y, curr_sample_weights)

            # calculate error and stump weight from weak learner prediction
            stump_pred = stump.predict(X)
            err = curr_sample_weights[(stump_pred != y)].sum()        
            
            if err > 0 and err < 1:
                stump_weight = np.log((1 - err) / err) / 2

                # update sample weights
                new_sample_weights = curr_sample_weights * np.exp(-stump_weight * y * stump_pred)
                new_sample_weights /= new_sample_weights.sum()
                
            else:
                if t>1: stump_weight = stump_weights[t-1]
                else: stump_weight = 1
                new_sample_weights = curr_sample_weights

            # If not final iteration, update sample weights for t+1
            if t+1 < self.__n_estimators: sample_weights[t+1] = (new_sample_weights)

            # save results of iteration
            if t>1:
                assert(stumps[t-1] != stump)
            stumps[t] = (stump)
            stump_weights[t] = (stump_weight)
            preds += (stump_weight * stump_pred)

            train_accs[t] = np.mean(np.sign(preds/(t+1)) == y)
            if self.__verbose: print(f"Iteration {t}: accuracy = {train_accs[t]}")

        self.__stump_weights = stump_weights
        self.__stumps = stumps
        self.__train_accs = train_accs
        return self