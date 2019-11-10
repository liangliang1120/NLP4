# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:23:04 2019

In this practical part, you will build a simple digits recognizer to check if the digit in the image is larger than 5. This assignmnet will guide you step by step to finish your first small project in this course

@author: us
"""
#1 - Packages
#sklearn is a famous package for machine learning.
#matplotlib is a common package for vasualization.

from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

#### 2 - Overvie of the dataset  
#    - a training set has m_train images labeled as 0 if the digit < 5 or 1 if the digit >= 5
#    - a test set contains m_test images labels as if the digit < 5 or 1 if the digit >= 5
#    - eah image if of shape (num_px, num_px ). Thus, each image is square(height=num_px and  width = num_px)
    

# Loading the data 
digits = datasets.load_digits()

# Vilizating the data
for i in range(1,11):
    plt.subplot(2,5,i)
    plt.imshow(digits.data[i-1].reshape([8,8]),cmap=plt.cm.gray_r)
    plt.text(3,10,str(digits.target[i-1]))
    plt.xticks([])
    plt.yticks([])
plt.show()

# Split the data into training set and test set 
X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.25)

# reformulate the label. 
# If the digit is smaller than 5, the label is 0.
# If the digit is larger than 5, the label is 1.

Y_train[Y_train < 5 ] = 0
Y_train[Y_train >= 5] = 1
Y_test[Y_test < 5] = 0
Y_test[Y_test >= 5] = 1

print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)


#3- Architecture of the neural network

#4 - Building the algorithm
#4.1- Activation function

def sigmoid(z):
    '''
    Compute the sigmoid of z
    Arguments: z -- a scalar or numpy array of any size.
    
    Return:
    s -- sigmoid(z)
    '''
    s = [ 1./(1 + np.exp(-1 * x)) for x in z]
    
    return s

# Test your code 
# The result should be [0.5 0.88079708]
print("sigmoid([0,2]) = " + str(sigmoid(np.array([0,2]))))

#4.2-Initializaing parameters
# Random innitialize the parameters

def initialize_parameters(dim):
    '''
    Argument: dim -- size of the w vector
    
    Returns:
    w -- initialized vector of shape (dim,1)
    b -- initializaed scalar
    '''
    
    w = np.random.randn(dim).reshape(-1,1)
    b = np.random.randn()
    
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    
    return w,b

#4.3-Forward and backward propagation
def propagate(w,b,X,Y):
    '''
    Implement the cost function and its gradient for the propagation
    
    Arguments:
    w - weights
    b - bias
    X - data
    Y - ground truth
    '''
    m = X.shape[1] # ?
    A = sigmoid(np.dot(X,w) + b)
    
    cost = -np.mean(np.dot(Y,np.log(A)))
    
    dw = np.dot((np.array(A).T-np.array(Y).T)[0].T,X).reshape(64,-1)/X.shape[0]
    db = np.mean((np.array(A).T-np.array(Y).T)[0].T)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {'dw':dw,
             'db':db}
    return grads, cost

#4.4 -Optimization


def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):
    '''
    This function optimize w and b by running a gradient descen algorithm
    Minimizing the cost function using gradient descent
    Arguments:
    w - weights
    b - bias
    X - data
    Y - ground truth
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps
    
    Returns:
    params - dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, 
    this will be used to plot the learning curve.
    
    '''
    
    costs = []
    
    for i in range(num_iterations):
        
        grads, cost = propagate(w,b,X,Y)
        
        dw = grads['dw']
        db = grads['db']
        
        w = w - dw*learning_rate
        b = b - db*learning_rate
        
        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
    
    params = {"w":w,
              "b":b}
    
    grads = {"dw":dw,
             "db":db}
    
    return params, grads, costs





def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w -- weights
    b -- bias 
    X -- data 
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[0]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[1],1)
    
    A = sigmoid(np.dot(X,w)+b)
    
    for i in range(len(A)):
        if A[i][0]<=0.5:
            Y_prediction[0][i] = 0
        else:
            Y_prediction[0][i] = 1
    
    assert(Y_prediction.shape == (1,m))
    
    return Y_prediction





def model(X_train, Y_train, X_test, Y_test, num_iterations, learning_rate,print_cost):
    """
    Build the logistic regression model by calling all the functions you have implemented.
    Arguments:
    X_train - training set
    Y_train - training label
    X_test - test set
    Y_test - test label
    num_iteration - hyperparameter representing the number of iterations to optimize the parameters
    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
    print_cost -- Set to true to print the cost every 100 iterations
    
    Returns:
    d - dictionary should contain following information w,b,training_accuracy, test_accuracy,cost
    eg: d = {"w":w,
             "b":b,
             "training_accuracy": traing_accuracy,
             "test_accuracy":test_accuracy,
             "cost":cost}
    """
    w,b = initialize_parameters(X_train.shape[1])

    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations,
                                        learning_rate, print_cost)
    Y_prediction = predict(w, b, X_train)
    traing_accuracy = 1-sum(abs(Y_train-Y_prediction)[0])/len(Y_train)
    Y_prediction_t = predict(w, b, X_test)    
    test_accuracy = 1-sum(abs(Y_test-Y_prediction_t)[0])/len(Y_test)
    cost = costs[-1]
    d = {"w":w,
             "b":b,
             "training_accuracy": traing_accuracy,
             "test_accuracy":test_accuracy,
             "cost":cost}
    return d


d = model(X_train, Y_train, X_test, Y_test, 3000, 0.02,True)