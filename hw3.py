#!/usr/bin/python3
# Homework 2 Code
import numpy as np
import pandas as pd
import sklearn.preprocessing as skp


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


def logistic_reg(X, y, w_init, max_its, eta, grad_threshold, regularization = None, lam = 0, debug = False):
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
        noup = 0
        if debug: print("w: ", w)

        loss, grad = cross_entropy_error(w, X, y)
        if debug: print("loss: ", loss, "grad: ", grad)
        #if t % 1000 == 0: print("loss: ", loss)
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
        if regularization == None:
            w = w - (eta*grad)

        elif regularization == "L2":
            if debug: print("regularization component", (2*eta*lam*w))
            w = w - ((eta*grad) + (2*eta*lam*w))
            #print("w: ", w.shape, "grad: ", grad.shape, "eta: ", eta, "lam: ", lam)

        elif regularization == "L1":
            wnoreg = w - (eta*grad)
            wregu = w - (eta*grad + (eta*lam*np.sign(w)))
            update = (eta*grad + (eta*lam*np.sign(w)))

            if debug:
                print("regularization component", wnoreg-wregu)

            for i in range(len(update)):
                if (np.sign(wnoreg[i]) != np.sign(wregu[i])) and wnoreg[i] != 0:
                    if debug: print("truncating")
                    update[i] = 0
                    noup += 1
            w = w - update
    #print("number of times truncation occured:", noup)
    e_in = cross_entropy_error(w, X, y)[0]



    return t, w, e_in


def main(max_its = 10**4, eta = 0.01, grad_threshold = 10**-6, normalize = True, reg = None, lam = 0, data = "hw3", debug = False):
    print("starting experiment using", reg, "regularization with lambda =", lam)

    if data == "hw3":
        data = np.load("digits_preprocess.npy", allow_pickle=True)
        X = data[0]
        X_test = data[1]
        y = data[2]
        y_test = data[3]

         #change the y labels to be -1 or 1
        y = np.where(y == 0, -1, 1)
        y_test = np.where(y_test == 0, -1, 1)
        #print(np.unique(y), np.unique(y_test))

        # drop X columns that are all the same value
        X = X[:,np.where(np.std(X, axis=0) != 0)[0]]
        X_test = X_test[:,np.where(np.std(X, axis=0) != 0)[0]]

    elif data == "hw2":
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

    elif data == "toy":
        X = np.array([[0, 1], [1, 0], [1, 1]])
        y = np.array([1, -1, -1])
        X_test = np.array([[1, 1]])
        y_test = np.array([-1])

    if normalize:
        scaler = skp.StandardScaler(with_mean=True, with_std=True)
        scaler.fit(X)
        X = scaler.transform(X)
        X_test = scaler.transform(X_test)
        # means = X.mean(axis=0)
        # stds = X.std(axis=0)
        # X = (X - means) / stds
        # X_test = (X_test - means) / stds

    w = np.zeros(X.shape[1]+1)

    t, w, e_in = logistic_reg(X, y, w, max_its, eta, grad_threshold, regularization=reg, lam=lam, debug=debug)

    #print("steps to finish: ", t+1)
    #print("E_in: ", e_in)
    #print("binary classification error (training data): ", find_binary_error(w, X, y))
    print("binary classification error (test data): ", find_binary_error(w, X_test, y_test))
    threshold = 1e-5
    print("number of 0s in the weight vector: ", np.sum(np.abs(w) < threshold))
    #print(w)

if __name__ == "__main__":
    main()
