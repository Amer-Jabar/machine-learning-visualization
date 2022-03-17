"""
    Helper methods for reusability
"""

from random import random, uniform
import numpy as np

def get_discrete_steps(data_range, scale):
    range_ = []
    for x in range(data_range):
        range_.append(x * scale)
    return range_

def get_random_data(data_range, x_rate, y_rate):
    data_x = []
    data_y = []
    for x in range(data_range):
        data_x.append(random() + (x * x_rate))
        value = uniform(1, -1) + (x * y_rate) + 1
        if value < 0:
            value *= -1
        data_y.append(value)
        
    return data_x, data_y

def get_random_coeffs():
    eta = 0.05
    w1 = 1.5
    w0 = 2

    return eta, w1, w0

def execute_gd(x, y, w1, w0, eta, x_, epochs, loss_hist, gradient_hist, w1_hist):
    if not loss_hist:
        loss_hist = []
    if not gradient_hist:
        gradient_hist = []
    if not w1_hist:
        w1_hist = []

    for iter_ in range(epochs):
        # y_hat = (X.w1) + w0
        # Matrix form of y = mx + b
        y_ = np.dot(w1, x_) + w0
        
        # MSE loss function
        loss = (1 / 100) * sum((y_ - y) ** 2)
        loss_hist.append(loss)
        
        # The gradient - slope - partial derivative of the linear equation (summation of linear equations)
        # We calculate as many gradient
        gradient_w0 = (1 / 100) * sum(((np.dot(w1, x) + w0) - y) * 1)
        gradient_w1 = (1 / 100) * sum(((np.dot(w1, x) + w0) - y) * x)
        gradient_hist.append(w1)
        
        # We subract the coeffecients by learning rate * gradients
        w1 = w1 - eta * gradient_w1
        w0 = w0 - eta * gradient_w0
        w1_hist.append(w1)

    return w1, w0, loss_hist, gradient_hist, w1_hist

