from django.http import JsonResponse
from django.shortcuts import render
from json import loads

import pandas as pd

from helpers.LinearRegressionHelper import get_discrete_steps, get_random_data, get_random_coeffs, execute_gd

# Create your views here.

def random_data(request):
    x_data, y_data = get_random_data(100, 1, 1)
    response = {
        'x': x_data,
        'y': y_data,
    }
    return JsonResponse(response)

def x_discrete_steps(request):
    x_steps = get_discrete_steps(100, 1)
    response = {
        'x_': x_steps,
    }
    return JsonResponse(response)

def generate_coeffs(request):
    eta, w1, w0 = get_random_coeffs()
    response = {
        'eta': eta,
        'w1': w1,
        'w0': w0
    }
    return JsonResponse(response)

def execute_algo(request):

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

    [w1, w0, loss_hist, gradient_hist, w1_hist] = execute_gd(
        x,
        y, 
        w1, 
        w0, 
        eta, 
        x_, 
        epochs, 
        loss_hist, 
        gradient_hist, 
        w1_hist
    )

    return JsonResponse({
        'w1': w1, 
        'w0': w0, 
        'loss_hist': loss_hist, 
        'gradient_hist': gradient_hist, 
        'w1_hist': w1_hist
    })

    return JsonResponse({ 'error': 'An error occured' })


