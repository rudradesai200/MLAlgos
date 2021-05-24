import numpy as np

class KernelLogistRegression:
    def __init__(self,kernel='rbf',lr=0.01,kernel_param=1,degree=2,max_iter=10,reg_param=0,verbose=False):
        """Initializes params for Kernel Logistic Regression

        Args:
            kernel (str, optional): Name of the kernel. Choice = ["linear","poly","rbf"]. Defaults to 'rbf'.
            lr (float, optional): Learning Rate. Defaults to 0.01.
            kernel_param (int, optional): Param used in Kernel. Defaults to 1.
            degree (int, optional): Will be used only in poly kernels. Defaults to 2.
            max_iter (int, optional): Maximum number of iterations.. Defaults to 10.
            reg_param (int, optional): Reg Param for l2 norm. Defaults to 1.
            verbose (bool, optional): Print extra details. Defaults to False.
        """
        self.__kernel_name = kernel
        self.__kernel_func = None
        self.__kernel_param = kernel_param
        self.__degree = degree
        self.__max_iter = max_iter
        self.__reg_param = reg_param
        self.__learning_rate = lr
        self.__verbose = verbose
        self.__alpha = None
        self.__Xtrain = None

        ## Setting the kernel_func
        if self.__kernel_name == "linear": self.__kernel_func = self.__linear_kernel
        elif self.__kernel_name == "poly": self.__kernel_func = self.__poly_kernel
        elif self.__kernel_name == "rbf": self.__kernel_func = self.__rbf_kernel
        else: raise ValueError(f"Invalid kernel name passed; {kernel}")

        # To be deleted after training
        self.__kernel_mat = None

        # Temporary stores used to reduce redundant caluclations
        self.__temp1 = None

    ## Kernel Functions
    def __linear_kernel(self,u,v) : return (np.dot(u,v))
    def __poly_kernel(self,u,v) : return ((self.__kernel_param + np.dot(u,v)) ** self.__degree)
    def __rbf_kernel(self,u,v) : return (np.exp((-self.__kernel_param)*(np.linalg.norm(u-v,axis=-1)**2)))
    
    ## Helpers
    def __sigmoid(x) : return (1/(1+np.exp(-x)))

    def __create_kernel_matrix(self,X):
        """Creates and stores the Kernel Matrix.

        Args:
            X (numpy array): Input Dataset
        """
        n,_ = X.shape
        self.__kernel_mat = np.zeros((n,n),np.float)
        for i in range(n):
            for j in range(i,n):
                ## here [i][j] == [j][i] because the implemented kernels are symmetric
                temp = self.__kernel_func(X[i],X[j])
                self.__kernel_mat[i][j] = temp
                self.__kernel_mat[j][i] = temp
        
   

    def __loss_function(self,Y,alpha):
        """
        Implements the logistic loss function using kernels.
        Optimized by vectorizing operations.
        Expect Y.shape = alpha.shape = (n,1)
        """
        main_loss = np.sum(np.logaddexp(0,-(Y * (self.__kernel_mat.T @ alpha))))
        reg_loss = (alpha.T @ self.__kernel_mat @ alpha) * self.__reg_param
        return main_loss + reg_loss

    def __grad_loss_function(self, Y, alpha):
        """
        Gradient of the logistic loss function.
        Optimized by vectorizing operationis.
        Expect Y.shape = alpha.shape = (n,1)
        """
        grad_loss = np.sum(self.__temp1 * KernelLogistRegression.__sigmoid(-((self.__kernel_mat.T @ alpha) * Y)), axis=0).reshape(-1,1)
        grad_reg_loss = (self.__kernel_mat.T @ alpha)*self.__reg_param
        return  grad_loss + grad_reg_loss


    def fit(self,X, y):
        """
        Fits the kernel logistic regression on the given dataset (X,y).
        Has one intended side effect of storing the X in the object.
        This X is used in prediction stage.

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
        # Basic checks
        (n,_) = X.shape
        y = y.reshape(-1,1)

        # Basic initializations
        self.__create_kernel_matrix(X)
        self.__temp1 = (-(self.__kernel_mat.T * y))
        self.__Xtrain = X
        alpha = np.ones((n,1),np.float)

        # gradient descent
        for iter in range(self.__max_iter):
            gradient = self.__grad_loss_function(y,alpha)
            alpha -= self.__learning_rate * gradient
            if self.__verbose:
                print(f"iteration {iter+1}/{self.__max_iter} : loss = {self.__loss_function(y,alpha)}")
        
        # Save some memory and exit
        self.__kernel_mat = None
        self.__temp1 = None
        self.__alpha = alpha

        return self

    def predict(self,X):
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

        # Helper function to process each row of test
        # Used to vectorise the prediction
        def get_pred(x):
            temp = self.__kernel_func(self.__Xtrain, x).reshape(1,-1) @ self.__alpha
            return np.sign(KernelLogistRegression.__sigmoid(temp) - 0.49)
        
        return np.apply_along_axis(get_pred,1,X).reshape(-1,1)


# How to use
if __name__ == "__main__":
    def load_dataset(loc):
        ds = np.load(loc)
        return (ds['arr_0'],ds['arr_1'],ds['arr_2'],ds['arr_3'])
    
    (X_train,Y_train,X_test,Y_test) = load_dataset("Data/dataset_C.npz")
    clf = KernelLogistRegression(lr=1,max_iter=100,kernel="linear",kernel_param=0.01,verbose=True)
    clf.fit(X_train,Y_train)
    Y_pred = clf.predict(X_test).reshape(-1,1)
    # print(Y_pred)
    # print(Y_test)
    print(np.mean(Y_pred == Y_test.reshape(-1,1)))
    