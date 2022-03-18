from django.http import HttpRequest, JsonResponse
from django.shortcuts import render
from json import loads

import pandas as pd

from helpers.LinearRegressionHelper import get_discrete_steps, get_random_data, get_random_coeffs, execute_gd

# Create your views here.

CORS_ORIGIN_HEADER = 'Access-Control-Allow-Origin'
CORS_ORIGIN_VALUE = 'http://127.0.0.1:4444'
CORS_METHOD_HEADER = 'Access-Control-Allow-Method'
CORS_METHOD_VALUE = 'GET'

def random_data(request):
    x_data, y_data = get_random_data(100, 1, 1)
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

def execute_algo(request):
    try:
        body = request.body
        x = loads(body)['x']
        y = loads(body)['y']
        w1 = loads(body)['w1']
        w0 = loads(body)['w0']
        eta = loads(body)['eta']
        x_ = loads(body)['x_']
        epochs = loads(body)['epochs']
        loss_hist = loads(body)['loss_hist']
        gradient_hist = loads(body)['gradient_hist']
        w1_hist = loads(body)['w1_hist']
    
        [w1, w0, loss_hist, gradient_hist, w1_hist] = execute_gd( x, y,  w1,  w0,  eta,  x_,  epochs,  loss_hist,  gradient_hist,  w1_hist)
    
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

    except Exception as err:
        return JsonResponse({
            'message': 'An error occured',
            'error': str(err)
        })


