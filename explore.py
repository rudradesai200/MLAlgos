import numpy as np
import matplotlib.pyplot as plt
from DecisionTreeContinous import DecisionTree

def load_dataset(loc):
    ds = np.load(loc)
    X_train = ds['arr_0']
    X_test = ds['arr_2']
    Y_train = ds['arr_1']
    Y_test = ds['arr_3']
    return (X_train,Y_train,X_test,Y_test)

def plot(plotter, DS="A"):
    (X_train,Y_train,_,_) = load_dataset(f"Data/dataset_{DS}.npz")
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

    Y = np.array(plotter(X)).reshape(n1,n2)
    ax = axs
    a = ax.contourf(X1,X2,Y,colors = ['#ffcccc','#99ffbb'])
    proxy = [plt.Rectangle((0,0),1,1,fc = pc.get_facecolor()[0]) for pc in a.collections]
    contour_legend = ax.legend(proxy[::-1],["classified as Y = 1","classified as Y = -1"])
    ax.add_artist(contour_legend)
    ax.set_ylabel(f"Dataset A")
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

(X_train,Y_train,X_test,Y_test) = load_dataset("Data/dataset_B.npz")
#DATASET A 
# tree = train_decision_tree(X_train,Y_train,20,-1,"accuracy")
#DATASET B 
# tree = train_decision_tree(X_train,Y_train,40,7,"accuracy")
#DATASET C 
# tree = train_decision_tree(X_train,Y_train,10,6,"accuracy")
#DATASET D 
# tree = train_decision_tree(X_train,Y_train,20,4,"accuracy")
# tree.root.print()
#DATASET A 
# tree = train_decision_tree(X_train,Y_train,20,-1,"entropy")
#DATASET B 
# tree = train_decision_tree(X_train,Y_train,40,7,"entropy")
#DATASET C 
# tree = train_decision_tree(X_train,Y_train,10,6,"entropy")
#DATASET D 
# tree = train_decision_tree(X_train,Y_train,20,4,"entropy")
# tree = train_decision_tree(X_train,Y_train,100,5,"entropy")

tree = DecisionTree(num_nodes_stop=20,max_level=-1)
tree.fit(X_train,Y_train)
Y_pred = tree.predict(X_test)

def plotter_tree(X): return tree.predict(X)
plot(plotter_tree,"B")

Y_pred = tree.predict(X_test)
print(f'{np.mean(Y_pred == Y_test)}')

# if __name__ == "__main__":
#     (X_train,Y_train,X_test,Y_test) = load_dataset("../Data/dataset_A.npz")
#     # limit = (3 * (len(X_train)))//4
#     limit = len(X_train)//4
#     incr = (limit)//(40)
#     # for level in range(1,6):
#     # for stop in range(2,limit,incr):
#     for nest in range(100,200,100):
#         # for level in range(1,8):
#         for stop in range(2, limit, incr):
#             forest = RandomForestClassifier(n_estimators=nest,criterion="entropy", min_samples_split=stop,max_features=0.5)
#             forest.fit(X_train,Y_train)
#             Y_pred = forest.predict(X_test)
#             print(f"nest - {nest} , level - inf, stop - {stop} , accuracy - {np.mean(Y_pred == Y_test)}")



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



