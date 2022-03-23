import numpy as np

def get_random_data():
    x = np.array([
        1, 2, 3, 4, 1, 2, 4, 3, 4, 5, 1, 2, 3, 4, 2, 3, 2, 3, 1, 4, 5, 6, 3, 1, 4, 1, 1, 2, 3, 5,
        12, 14, 13, 10, 11, 12, 11, 10, 13, 14, 12, 10, 10, 12, 10, 13, 12, 14, 12, 11, 10, 10, 14, 12, 13, 10, 11, 10, 14, 14,
    ])
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    return x, y

def get_random_coeffs():
    w1 = 0
    w0 = 0
    eta = 0.01
    return w1, w0, eta

def sigmoid(z):
    z = z.astype('float64')
    return 1.0 / (1 + np.exp(-z))

def predict(x, w1, w0):
    if str(type(x)) == "<class 'int'>" or str(type(x)) == "<class 'float'>":
        raise TypeError('Type must be a list-like')
    if type(x) != 'numpy.ndarray':
        x = np.array(x)
        
    return sigmoid(w1 * x + w0)

def cross_entropy(data_size, y, pred):
    if type(y) != "<class 'numpy.ndarray'>":
        y = np.array(y)
    if type(pred) != "<class 'numpy.ndarray'>":
        pred = np.array(pred)

    part_a = (y * np.log(pred))
    part_b = ((1 - y) * np.log(1 - pred))
    
    cost = (-1 / data_size) * sum(part_a + part_b)
    return cost
    
def gradient(x, y, pred, data_size):
    return np.dot(x, (pred - y)) / data_size
    
def update(weight, eta, gradient):
    gradient *= eta
    weight -= gradient
    return weight

def execute_lr(x, y, data_size, eta, w1, w0, loss_hist = []):    
    pred = predict(x, w1, w0)
    cost = cross_entropy(data_size, y, pred)
    loss_hist.append(cost)
    slope = gradient(x, y, pred, data_size)
    w1 = update(w1, eta, slope)
    w0 = update(w0, eta, slope)
        
    return w1, w0, loss_hist

