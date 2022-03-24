import numpy as np

def replace_one(arr, replacer_value):
    for i in range(len(arr)):
        if arr[i] == 1:
            arr[i] = replacer_value

def fill_with_values(size, value_1, value_2):
    array = []
    for x in range(size):
        if x < size / 2:
            array.append(value_1)
        else:
            array.append(value_2)
    return array

def generate_random_data(size, lower_bound, upper_bound):
    arr = []
    for i in range(size):
        if i < size / 2:
            arr.append((np.random.rand() * upper_bound))
        else:
            arr.append((np.random.rand() * upper_bound) + lower_bound / 2)
    return arr

def get_random_data():
    x = generate_random_data(100, 10, 10)
    y = fill_with_values(len(x), 0, 1)
    return x, y

def get_random_coeffs():
    w1 = np.random.randint(0, 3)
    w0 = np.random.randint(0, 3)
    eta = 0.01
    if w1 == 0 and w0 >= 0:
        w1 = 1
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

    # The problem encounterd here is log of 0 which is shown as division by 0
    # The solution is to change the value in order not to become 0
    replace_one(pred, 0.999)
    part_a = (y * np.log(pred))
    part_b = ((1 - y) * np.log(1 - pred))
    sum_ = sum(part_a + part_b)
    cost = (-1 / data_size) * sum_
    return cost
    
def gradient(x, y, pred, data_size):
    return np.dot(x, (pred - y)) / data_size
    
def update(weight, eta, gradient):
    gradient *= eta
    weight -= gradient
    return weight

def execute_lr(x, y, data_size, eta, w1, w0, loss_hist):        
    pred = predict(x, w1, w0)
    cost = cross_entropy(data_size, y, pred)
    loss_hist.append(cost)
    slope = gradient(x, y, pred, data_size)
    w1 = update(w1, eta, slope)
    w0 = update(w0, eta, slope)

    return w1, w0, loss_hist, list(pred)


