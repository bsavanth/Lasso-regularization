import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
alpha = 0.01
epsilon = 0.001
lamda = 5

def partition():
    ftrain = open("training.txt", "w+")
    ftest = open("testing.txt", "w+")
    count = 0
    f = open('input.csv', 'r+')
    for line in f:
        if (count < 48):
            ftrain.write(line)
            count += 1
        else:
            ftest.write(line)
            count += 1
    f.close()
    ftrain.close()
    ftest.close()

def Loss(X,Y, theta):

  global alpha, epsilon, lamda
  cost = ((Y.T-theta@X.T)@(Y.T-theta@X.T).T)/(2*len(X))
  regparam = np.power(theta,2)
  return  (cost+lamda*np.sum(regparam)/(2*len(X))).item()

def Gradient(X,Y,theta,j):
  global alpha, epsilon, lamda
  Xnew = X[:, j]
  Xnew = Xnew.reshape(len(X), 1)
  temp = ((X @ theta.T)-Y)*Xnew
  return (np.sum(temp)/len(X))+(lamda*theta[0][j]/len(X))

def LRwithQR(X, Y, theta):
    global alpha, epsilon, lamda
    loss = []
    k = 0
    while(not Convergence(k,loss)):

        for j in range(16):
          theta[0][j] = theta[0][j] - alpha*Gradient(X, Y, theta, j)
        
        loss.append(Loss(X, Y, theta))
          
        k += 1
    
    return theta, loss, k

def Convergence(k,cost):
    global alpha, epsilon, lamda
    if (k <2):
        return False
    criteria = (abs(cost[k - 2] - cost[k-1]) * 100) / cost[k - 2]
    if(criteria<epsilon):
        return True
    else:
        return False


def SquaredError(X,Y,theta):
  
    error = np.power(((X @ theta.T)-Y), 2)
    
    return np.sum(error)/(2 * len(X))

def Round(theta):
  zeroparams = 0
  for i in range(16):
      if abs(theta[0][i])<0.01:
          theta[0][i] = 0
          zeroparams += 1
  return zeroparams
def Normalize(data):
    data = (data - data.min()) / (data.max()-data.min())
    return data


def main():
    partition()
    data = pd.read_csv('training.txt',
                          names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
                                 "x14", "x15", "y"])
    data = Normalize(data)
    X = data.iloc[:, 0:15]
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)
    Y = data.iloc[:, 15:16].values
    theta = np.zeros([1, 16])
    theta, cost, iterations = LRwithQR(X, Y, theta)
    zero_parameters=Round(theta)
    print("\nFinal updated parameters:","\n")
    print(theta[0])
    print(zero_parameters, "zero parameters found\n")

    data = pd.read_csv('testing.txt',
                            names=["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13",
                                   "x14", "x15", "y"])
    data = Normalize(data)
    X = data.iloc[:, 0:15]
    ones = np.ones([X.shape[0], 1])
    X = np.concatenate((ones, X), axis=1)
    Y = data.iloc[:, 15:16].values
    print("Squared loss with testing data:",SquaredError(X, Y, theta))
    print("Number of iterations it took for the loss function to converge:",iterations)
    fig, ax = plt.subplots()
    ax.plot(np.arange(iterations), cost, 'r')
    ax.set_xlabel('number of iterations')
    ax.set_ylabel('Loss function')
    ax.set_title('Loss function vs number of iterations')
    plt.show()

main()

