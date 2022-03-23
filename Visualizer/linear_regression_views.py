from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from json import loads

import pandas as pd

from helpers.LinearRegressionHelper import get_discrete_steps, get_random_data, get_random_coeffs, execute_gd

# Create your views here.

CORS_ORIGIN_HEADER = 'Access-Control-Allow-Origin'
CORS_ORIGIN_VALUE = '*'
CORS_METHOD_HEADER = 'Access-Control-Allow-Method'
CORS_METHOD_VALUE = '*'

DATA_SIZE = 100
X_SCALE = 1
Y_SCALE = 1
DATA_SIZE_RANGE_ = 10

def base_endpoint(request):
    responseBody = { 'message': 'You have reached the base endpoint' }
    response = JsonResponse(responseBody)
    response.headers[CORS_ORIGIN_HEADER] = CORS_ORIGIN_VALUE
    response.headers[CORS_METHOD_HEADER] = CORS_METHOD_VALUE
    return response

def random_data(request):
    x_data, y_data = get_random_data(DATA_SIZE, X_SCALE, Y_SCALE, DATA_SIZE_RANGE_)
    responseBody = {
        'x': x_data,
        'y': y_data,
    }
    response = JsonResponse(responseBody)
    response.headers[CORS_ORIGIN_HEADER] = CORS_ORIGIN_VALUE
    response.headers[CORS_METHOD_HEADER] = CORS_METHOD_VALUE
    return response

def x_discrete_steps(request):
    x_steps = get_discrete_steps(100, 1)
    responseBody = {
        'x_': x_steps,
    }
    response = JsonResponse(responseBody)
    response.headers[CORS_ORIGIN_HEADER] = CORS_ORIGIN_VALUE
    response.headers[CORS_METHOD_HEADER] = CORS_METHOD_VALUE
    return response

def generate_coeffs(request):
    eta, w1, w0 = get_random_coeffs()
    responseBody = {
        'eta': eta,
        'w1': w1,
        'w0': w0
    }
    response = JsonResponse(responseBody)
    response.headers[CORS_ORIGIN_HEADER] = CORS_ORIGIN_VALUE
    response.headers[CORS_METHOD_HEADER] = CORS_METHOD_VALUE
    return response

@csrf_exempt
def execute_algo(request):
    body = request.body
    x = loads(body)['x']
    y = loads(body)['y']
    w1 = loads(body)['w1']
    w0 = loads(body)['w0']
    eta = loads(body)['eta']
    x_ = loads(body)['x_']
    epochs = loads(body)['epochs']

    [w1, w0, loss_hist, gradient_hist, w1_hist] = execute_gd(x, y,  w1,  w0,  eta,  x_,  epochs)
    responseBody = {
        'w1': w1, 
        'w0': w0, 
        'loss_hist': loss_hist, 
        'gradient_hist': gradient_hist, 
        'w1_hist': w1_hist
    }

    response = JsonResponse(responseBody)
    response.headers[CORS_ORIGIN_HEADER] = CORS_ORIGIN_VALUE
    response.headers[CORS_METHOD_HEADER] = CORS_METHOD_VALUE
    return response


