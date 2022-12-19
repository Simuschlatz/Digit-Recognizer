import numpy as np
import pandas as pd

data = pd.read_csv("Data/training_data.csv")
# m = height, n = width
m, n = data.shape
Data  = np.array(data)
np.random.shuffle(Data)

Data_train = Data.T
Labels = Data_train[0]
Inpt_layer = Data_train[1:n]
Inpt_layer = Inpt_layer / 255.

Iterations_training = 200

def init_params():
    W1 = np.random.rand(80, 784) - 0.5
    b1 = np.random.rand(80, 1) - 0.5
    W2 = np.random.rand(10, 80) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(0, Z)

def softmax(Z):
    # each col divided by the sum of the matrix with the cols collapsed into one row
    return np.exp(Z) / sum(np.exp(Z))

def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1 # Hidden Layer
    A1 = ReLU(Z1) # Activation Func Rectified Linear-Unit
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2) # Probabilities
    return Z1, A1, Z2, A2

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, 10))
    # np.arange(Y.size) creates an array of the indices for each training example
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def deriv_ReLU(Z):
    return Z > 0

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T) # 10 x 10
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * deriv_ReLU(Z1)
    dW1 = 1 / m * dZ1.dot(X.T) # 10 x 10
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1    
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2   
    return W1, b1, W2, b2

def make_prediction(W1, b1, W2, b2, X):
    _, _, _, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

def get_predictions(A2):
    return np.argmax(A2, 0)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size

def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if not i % 20:
            print(f"Iteration: {i}")
            predictions = get_predictions(A2)
            print(f"ACCURACY: {get_accuracy(predictions, Y)}")
    return W1, b1, W2, b2

def get_trained_weights():
    W1t, b1t, W2t, b2t = gradient_descent(Inpt_layer, Labels, 1, Iterations_training)
    return W1t, b1t, W2t, b2t

