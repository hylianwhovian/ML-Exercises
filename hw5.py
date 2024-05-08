#!/usr/bin/python3
# Homework 5 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt

def adaboost_trees(X_train, y_train, X_test, y_test, n_trees):
    # %AdaBoost: Implement AdaBoost using decision trees
    # %   using decision stumps as the weak learners.
    # %   X_train: Training set
    # %   y_train: Training set labels
    # %   X_test: Testing set
    # %   y_test: Testing set labels
    # %   n_trees: The number of trees to use

    # train_error will be a list containing the training error of each iteration, same for test_error

    # step 1: initialize weights
    # they should be uniform at the start and sum to 1
    # also initialize an empty list of stumps and alphas and train_error and test_error

    weights = np.ones(len(X_train))/len(X_train)
    stumps = []
    alphas = []
    train_errors = []
    test_errors = []

    # step 2: iteratively train decision stumps

    for step in range(n_trees):
        # train a decision stump
        
        stump = DecisionTreeClassifier(max_depth=1, criterion="entropy").fit(X_train, y_train, sample_weight=weights)
        stumps.append(stump)

        # calculate the in-sample error of the stump

        stump_preds = stump.predict(X_train)
        epsilon = np.sum(weights[stump_preds != y_train]) / len(y_train)
        if epsilon > 1 or epsilon < 0: print("epsilon is not between 0 and 1")

        # calculate the gamma of the stump

        gamma = np.sqrt((1 - epsilon) / epsilon)

        # calculate the Z of the stump

        Z = gamma*epsilon + (1/gamma)*(1 - epsilon)

        # calculate the alpha of the stump

        alpha = 0.5 * np.log((1 - epsilon) / epsilon)
        alphas.append(alpha)
        shaped_alphas = np.array(alphas).reshape(-1, 1)

        # update the weights

        # print("step: ", step)
        # print("epsilon: ", epsilon)
        # print("gamma: ", gamma)
        # print("Z: ", Z)
        # print("alpha: ", alpha)
        # print("weights: ", weights)

        weights = (1/Z) * weights * np.exp(-alpha * y_train * stump_preds)

        # calculate the training error of the current ensemble learner

        train_preds = np.array([stumps[i].predict(X_train) for i in range(step + 1)])
        # print(train_preds)
        train_preds = np.sign(np.sum(shaped_alphas*train_preds, axis = 0))

        train_error = np.sum(train_preds != y_train) / len(y_train)
        train_errors.append(train_error)

        # calculate the test error of the current ensemble learner

        test_preds = np.array([stumps[i].predict(X_test) for i in range(step + 1)])
        test_preds = np.sign(np.sum(shaped_alphas*test_preds, axis = 0))

        test_error = np.sum(test_preds != y_test) / len(y_test)
        test_errors.append(test_error)

        # print("on step:", step, "the model made", np.sum(test_preds != y_test), "errors out of", len(y_test), "test points", "for a test error of", test_error)

    return train_errors, test_errors


def main_hw5(problem = "1v3", num_trees = 200):
    # Load data
    og_train_data = np.genfromtxt('zip.train')
    og_test_data = np.genfromtxt('zip.test')

    # Set up data
    # the target values are the first column
    X_train = og_train_data[:,1:]
    y_train = og_train_data[:,0]
    X_test = og_test_data[:,1:]
    y_test = og_test_data[:,0]

    # now, we want to create two datasets: one with only 1s and 3s, and one with only 3s and 5s

    X_train_1v3 = X_train[(y_train == 1) | (y_train == 3)]
    y_train_1v3 = y_train[(y_train == 1) | (y_train == 3)]
    X_test_1v3 = X_test[(y_test == 1) | (y_test == 3)]
    y_test_1v3 = y_test[(y_test == 1) | (y_test == 3)]

    X_train_3v5 = X_train[(y_train == 3) | (y_train == 5)]
    y_train_3v5 = y_train[(y_train == 3) | (y_train == 5)]
    X_test_3v5 = X_test[(y_test == 3) | (y_test == 5)]
    y_test_3v5 = y_test[(y_test == 3) | (y_test == 5)]

    # we nee to change the labels to -1 and 1

    y_train_1v3[y_train_1v3 == 1] = -1
    y_train_1v3[y_train_1v3 == 3] = 1
    y_test_1v3[y_test_1v3 == 1] = -1
    y_test_1v3[y_test_1v3 == 3] = 1

    y_train_3v5[y_train_3v5 == 3] = -1
    y_train_3v5[y_train_3v5 == 5] = 1
    y_test_3v5[y_test_3v5 == 3] = -1
    y_test_3v5[y_test_3v5 == 5] = 1

    if problem == "1v3":
        X_train = X_train_1v3
        y_train = y_train_1v3
        X_test = X_test_1v3
        y_test = y_test_1v3

    elif problem == "3v5":
        X_train = X_train_3v5
        y_train = y_train_3v5
        X_test = X_test_3v5
        y_test = y_test_3v5

    train_error, test_error = adaboost_trees(X_train, y_train, X_test, y_test, num_trees)

    return train_error, test_error


if __name__ == "__main__":
    main_hw5()
