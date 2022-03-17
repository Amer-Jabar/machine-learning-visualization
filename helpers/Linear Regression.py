"""
    This project is the simple linear regression algorithm made in python from scratch.
    It depends on libraries like matplotlib and numpy. There are comments with each step
    to demonstrate the mathematis behind the algorithm. The plotting scripts show
    the optimization steps and how Gradient Descent reduces the loss function.


    Current dir: E:\CS Projects\Machine Learning Projects\Learning Projects\Regression\Linear Regression - Scratch
"""

# Setting base dir
import os
os.chdir('E://CS Projects/Machine Learning Projects/Learning Projects/Regression/Linear Regression - Scratch')

# Importing relevant libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from LinearRegressionHelper import get_discrete_steps, get_random_data

# Generating random data to simulate linear regression
x, y = get_random_data(100, 0.1, 0.05);
x = pd.DataFrame(x).iloc[:, 0]
y = pd.DataFrame(y).iloc[:, 0]

# Generate x axis data for straight line simulation
x_ = pd.DataFrame(get_discrete_steps(100, 0.1)).iloc[:, 0]

# eta = learning rate
# w_sub_i = coeffecients
eta = 0.05
w1 = 1.5
w0 = 2

# Storing step information
loss_hist = []
gradient_hist = []
w1_hist = []

for iter_ in range(10000):
    # y_hat = (X.w1) + w0
    # Matrix form of y = mx + b
    y_ = np.dot(w1, x_) + w0
    
    # MSE loss function
    loss = (1 / 100) * sum((y_ - y) ** 2)
    loss_hist.append(loss)
    
    # The gradient - slope - partial derivative of the linear equation (summation of linear equations)
    # We calculate as many gradient
    gradient_w0 = (1 / 100) * sum(((w1 * x + w0) - y) * 1)
    gradient_w1 = (1 / 100) * sum(((w1 * x + w0) - y) * x)
    gradient_hist.append(w1)
    
    # We subract the coeffecients by learning rate * gradients
    w1 = w1 - eta * gradient_w1
    w0 = w0 - eta * gradient_w0
    w1_hist.append(w1)


# Plotting everything
fig, axes = plt.subplots(1, 3)

axes[0].scatter(x, y)
axes[0].plot(x_, y_, 'r')
axes[1].plot(w1_hist, loss_hist, 'g')
axes[2].plot(gradient_hist, loss_hist, 'r')



# Mathematical notes:
#    1.Direction of the gradient is where the algorithm oppositely goes.
#    2.Choosing a large eta will oscillate until convergence.
#    3.Choosing a very large eta will make the algorithm diverge.
#    4.The selected method is netwon.
    
    
    
    
    

