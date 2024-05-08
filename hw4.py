#!/usr/bin/python3
# Homework 4 Code
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statistics import mode




def bagged_trees(X_train, y_train, X_test, y_test, num_bags):
    # The `bagged_tree` function learns an ensemble of numBags decision trees 
    # and also plots the  out-of-bag error as a function of the number of bags
    #
    # % Inputs:
    # % * `X_train` is the training data
    # % * `y_train` are the training labels
    # % * `X_test` is the testing data
    # % * `y_test` are the testing labels
    # % * `num_bags` is the number of trees to learn in the ensemble
    #
    # % Outputs:
    # % * `out_of_bag_error` is the out-of-bag classification error of the final learned ensemble
    # % * `test_error` is the classification error of the final learned ensemble on test data
    #
    # % Note: You may use sklearns 'DecisonTreeClassifier'
    # but **not** 'RandomForestClassifier' or any other bagging function

    out_of_bag_error = 0
    test_error = 0

    # first step: create our bagged datasets
    # I will do this by sampling with replacement indices from the training data, then using those indices to create a new dataset

    # create a list of indices
    indices = np.arange(len(X_train))

    # create a list of indices to be used for each bag
    bag_indices = np.random.choice(indices, size = (num_bags, len(X_train)), replace = True)

    # now we create a list of models, each trained from a different bag
    models = [DecisionTreeClassifier(criterion="entropy").fit(X_train[bag_indices[i]], y_train[bag_indices[i]]) for i in range(num_bags)]

    # next, we calculate the test error for the aggregate model
    # the way we aggregate the model is simply by taking the mode of the predictions of each model

    preds = np.array([models[i].predict(X_test) for i in range(num_bags)])
    agg_preds = stats.mode(preds)[0].flatten()

    # now that we have the aggregate predictions, we can calculate easily calculate the test error (binary error)

    test_error = np.sum(agg_preds != y_test) / len(y_test)

    # now we calculate the out of bag error
    # the first step is to create a list of bags for which each data point is not in the bag
    # so it will be a list of lists, where each list represents the bags for which its corresponding data point is not in any of those bags

    oob_ids = [[i for i in range(num_bags) if j not in bag_indices[i]] for j in range(len(X_train))] 

    # now, for each data point, we calculate the error of the aggregate model on the bags for which it is not in
    oob_preds = [
        [
            models[i].predict(X_train[j].reshape(1, -1))[0] for i in oob_ids[j] if len(oob_ids[j]) > 0
        ] for j in range(len(oob_ids))
    ]

    
    # aggregate the predictions
    oob_agg_preds = [mode(oob_pred) for oob_pred in oob_preds if len(oob_pred) > 0]

    # calculate the oob error
    oob_y_train = y_train[[i for i in range(len(oob_ids)) if len(oob_ids[i]) > 0]]

    out_of_bag_error = np.sum(oob_agg_preds != oob_y_train) / len(oob_y_train)

    print("with " + str(num_bags) + " bags, the out of bag error is " + str(out_of_bag_error) + " and the test error is " + str(test_error))

    return out_of_bag_error, test_error


# this just trains a single decision tree and calculates the test/train error
def single_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(criterion="entropy").fit(X_train, y_train)
    preds = model.predict(X_test)
    test_error = np.sum(preds != y_test) / len(y_test)
    preds = model.predict(X_train)
    train_error = np.sum(preds != y_train) / len(y_train)
    return train_error, test_error


def main_hw4(problem = "1v3"):

    num_bags =0

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

    bag_errors = []
    bags = np.arange(1, 201)
    for bag in bags:
        out_of_bag_error, test_error = bagged_trees(X_train, y_train, X_test, y_test, bag)
        bag_errors.append([out_of_bag_error, test_error])

    train_error, test_error = single_decision_tree(X_train, y_train, X_test, y_test)

    return bag_errors, test_error


if __name__ == "__main__":
    main_hw4()

