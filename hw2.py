#!/usr/bin/python3
# Homework 2 Code
import numpy as np
import pandas as pd


def find_binary_error(w, X, y):
    # find_binary_error: compute the binary error of a linear classifier w on data set (X, y)
    # Inputs:
    #        w: weight vector
    #        X: data matrix (without an initial column of 1s)
    #        y: data labels (plus or minus 1)
    # Outputs:
    #        binary_error: binary classification error of w on the data set (X, y)
    #           this should be between 0 and 1.

    # Your code here, assign the proper value to binary_error:
    ones = np.ones((X.shape[0],1))
    X = np.hstack((ones, X))
    outputs = X @ w.T
    sigmoids = 1 / (1 + np.exp(-outputs))
    preds = np.where(sigmoids > 0.5, 1, -1)
    corrects = np.where(preds == y, 1, 0)
    binary_error = 1 - corrects.sum() / len(corrects)
    return binary_error

def cross_entropy_error(w, X, y):
    # for this function, assume X already has the ones column

    # find E_in
    outputs = (X @ w.T) * -y
    outputs2 = np.log(1 + np.exp(outputs))
    E_in = np.mean(outputs2)

    # find the gradient
    top = np.multiply(X, y.reshape(-1,1))
    bottom = (1+np.exp((X @ w.T)*y)).reshape(-1,1)
    grad = -np.mean(top/bottom, axis=0)
    return E_in, grad


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold):
    # logistic_reg learn logistic regression model using gradient descent
    # Inputs:
    #        X : data matrix (without an initial column of 1s)
    #        y : data labels (plus or minus 1)
    #        w_init: initial value of the w vector (d+1 dimensional)
    #        max_its: maximum number of iterations to run for
    #        eta: learning rate
    #        grad_threshold: one of the terminate conditions; 
    #               terminate if the magnitude of every element of gradient is smaller than grad_threshold
    # Outputs:
    #        t : number of iterations gradient descent ran for
    #        w : weight vector
    #        e_in : in-sample error (the cross-entropy error as defined in LFD)

    # Your code here, assign the proper values to t, w, and e_in:

    # step 1: add ones column to X
    ones = np.ones((X.shape[0],1))
    X = np.hstack((ones, X))

    # step 2: initialize w
    w = w_init

    # step 3: do gradient descent
    for t in range(max_its):
        # compute gradient
        loss, grad = cross_entropy_error(w, X, y)
        h = 1e-5
        # gradient = np.zeros(w.shape)
        # for i in range(len(w)):
        #     w[i] += h
        #     loss2, grad2 = cross_entropy_error(w, X, y)
        #     gradient[i] = (loss2 - loss) / h
        #     w[i] -= h
        # check terminate condition
        if np.linalg.norm(grad) < grad_threshold:
            print("breaking")
            break

        # update w
        w = w - (eta*np.linalg.norm(grad)) * grad

    e_in = cross_entropy_error(w, X, y)[0]



    return t, w, e_in


def main(max_its = 10**4, eta = 10**-5, grad_threshold = 10**-3, normalize = False):
    # Load training data
    train_data = pd.read_csv('clevelandtrain.csv')

    # Load test data
    test_data = pd.read_csv('clevelandtest.csv')

    # Your code here

    X = train_data[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]].to_numpy()
    y = train_data["heartdisease::category|0|1"].to_numpy()
    y = np.where(y == 0, -1, 1)
    y_test = test_data["heartdisease::category|0|1"].to_numpy()
    y_test = np.where(y_test == 0, -1, 1)
    X_test = test_data[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]].to_numpy()

    if normalize:
        means = X.mean(axis=0)
        stds = X.std(axis=0)
        X = (X - means) / stds
        X_test = (X_test - means) / stds

    w = np.zeros(X.shape[1]+1)

    t, w, e_in = logistic_reg(X, y, w, max_its, eta, grad_threshold)

    print("steps to finish: ", t+1)
    print("E_in: ", e_in)
    print("binary classification error (training data): ", find_binary_error(w, X, y))
    print("binary classification error (test data): ", find_binary_error(w, X_test, y_test))

if __name__ == "__main__":
    main()
