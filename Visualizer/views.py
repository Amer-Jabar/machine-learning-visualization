from django.http import JsonResponse
from django.shortcuts import render

import pandas as pd

from helpers.LinearRegressionHelper import get_discrete_steps, get_random_data

# Create your views here.

def random_data(request):
    x_data, y_data = get_random_data(100, 1, 1)
    response = {
        'x_data': x_data,
        'y_data': y_data,
    }
    return JsonResponse(response)



