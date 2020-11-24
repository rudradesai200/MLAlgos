import numpy as np

### Helper functions
def sigmoid(x) : return (1/(1+np.exp(-x)))
def linear_kernel(u,v,kernel_param) : return (np.dot(u,v))
def poly_kernel(u,v,kernel_param) : return ((1 + np.dot(u,v)) ** kernel_param)
def rbf_kernel(u,v,kernel_param) : return (np.exp((-kernel_param)*(np.linalg.norm(u-v)**2)))
def rescale_X(X): (X - np.min(X,axis=0))/(np.max(X,axis=0) - np.min(X,axis=0))

### Useful lists
kernel_name = ["linear","poly","rbf"]
kernel_funcs = [linear_kernel,poly_kernel,rbf_kernel]    

def get_kernel_mat(kernel, X, kernel_param): return np.array([kernel_funcs[kernel](X[i],X[j],kernel_param) for i in range(len(X)) for j in range(len(X))]).reshape(len(X),len(X))
def loss_function(K, Y,alpha,reg_param): return np.sum(np.log(1 + np.exp(-(Y * (K.T @ alpha))))) + (reg_param)*(alpha.T @ K @ alpha)
def grad_loss_function(K, Y,alpha,reg_param): return (K.T @ alpha)*reg_param + np.sum((-(K.T * Y)) * sigmoid(-((K.T @ alpha) * Y)), axis=0).reshape(-1,1)


def train_logistic_regression(X, Y, kernel=0, reg_param=0., kernel_param=1., num_iter=10):
    # Basic checks
    (n,d) = X.shape
    Y = Y.reshape(-1,1)

    # initializations
    kernel_mat = get_kernel_mat(kernel, X, kernel_param)
    alpha = np.ones((n,1),np.float)
    eta = 0.01

    # gradient descent
    for i in range(num_iter):
        temp = grad_loss_function(kernel_mat,Y,alpha,reg_param)
        alpha -= eta * temp
        # print(f"iteration {i+1}/{num_iter} : loss = {loss_function(kernel_mat,Y,alpha,reg_param)}")
    
    return alpha

def test_logistic_regression(X_train, Y_train, X_test, kernel, kernel_param, reg_param):
    kernel_idx = np.argmax(kernel == np.array(kernel_name))
    
    alpha = train_logistic_regression(X_train, Y_train, kernel_idx, reg_param, kernel_param)

    def get_pred(x):
        temp = 0.0
        for i in range(len(X_train)):
            temp += (alpha[i]*kernel_funcs[kernel_idx](X_train[i],x,kernel_param))
        return np.sign(sigmoid(temp) - 0.5)
    
    Ypred = np.apply_along_axis(get_pred,1,X_test)
    return Ypred