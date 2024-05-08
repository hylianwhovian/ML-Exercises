#!/usr/bin/python3
# Homework 1 Code
import numpy as np
import matplotlib.pyplot as plt


def perceptron_learn(data_in):
    # Run PLA on the input data
    #
    # Inputs: data_in: Assumed to be a matrix with each row representing an
    #                (x,y) pair, with the x vector augmented with an
    #                initial 1 (i.e., x_0), and the label (y) in the last column
    # Outputs: w: A weight vector (should linearly separate the data if it is linearly separable)
    #        iterations: The number of iterations the algorithm ran for

    # Your code here, assign the proper values to w and iterations:
    
    # First, we initialize w with random weights inbetween 0 and 1
    w = np.random.rand(data_in.shape[1]-1)

    # next, we set the value of w0 to 0
    w[0] = 0

    # initialize the number of iterations to 0
    iterations = 0

    # run PLA
    not_done = True
    while not_done:
        prev = w
        for pair in data_in:
            if np.sign(np.dot(w, pair[0:-1])) != pair[-1]:
                w = w + pair[-1] * pair[0:-1]
                iterations += 1
        if np.array_equal(w, prev):
            not_done = False


    return w, iterations





def perceptron_experiment(N, d, num_exp):
    # Code for running the perceptron experiment in HW1
    # Implement the dataset construction and call perceptron_learn; repeat num_exp times
    #
    # Inputs: N is the number of training data points
    #         d is the dimensionality of each data point (before adding x_0)
    #         num_exp is the number of times to repeat the experiment
    # Outputs: num_iters is the # of iterations PLA takes for each experiment
    #          bounds_minus_ni is the difference between the theoretical bound and the actual number of iterations
    # (both the outputs should be num_exp long)

    # Initialize the return variables
    num_iters = np.zeros((num_exp,))
    bounds_minus_ni = np.zeros((num_exp,))

    # Your code here, assign the values to num_iters and bounds_minus_ni:

    for i in range(num_exp):
        # create the optimal separator
        w_star = np.random.rand(d+1)
        w_star[0] = 0

        # generate the random training set
        x = np.random.uniform(-1, 1, (N, d+1))
        x[:, 0] = 1

        # calculate the labels
        y = np.sign(np.dot(x, w_star))

        # combine the data and labels
        data_in = np.column_stack((x, y))

        # calculate the theoretical bound
        R = np.max(np.linalg.norm(x, axis=1))
        rho = np.min(np.dot(x, w_star) * y)
        bound = ((R**2)*(np.linalg.norm(w_star)**2))/(rho**2)

        # run PLA
        w, iterations = perceptron_learn(data_in)

        # update num_iters and bounds_minus_ni
        num_iters[i] = iterations
        bounds_minus_ni[i] = bound - iterations

    return num_iters, bounds_minus_ni


def main():
    print("Running the experiment...")
    num_iters, bounds_minus_ni = perceptron_experiment(100, 10, 1000)

    print("Printing histogram...")
    plt.hist(num_iters)
    plt.title("Histogram of Number of Iterations")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Count")
    plt.show()

    print("Printing second histogram")
    plt.hist(np.log(bounds_minus_ni))
    plt.title("Bounds Minus Iterations")
    plt.xlabel("Log Difference of Theoretical Bounds and Actual # Iterations")
    plt.ylabel("Count")
    plt.show()

if __name__ == "__main__":
    main()
