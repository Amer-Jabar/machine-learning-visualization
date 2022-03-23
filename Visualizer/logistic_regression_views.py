from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from json import loads

from helpers.LogisticRegressionHelper import get_random_data, get_random_coeffs, execute_lr

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
    x_data, y_data = get_random_data()
    responseBody = {
        'x': x_data,
        'y': y_data,
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

    w1, w0, loss_hist = execute_lr(x, y, len(x), eta, w1, w0)

    responseBody = {
        'w1': w1, 
        'w0': w0, 
        'loss_hist': loss_hist, 
    }

    response = JsonResponse(responseBody)
    response.headers[CORS_ORIGIN_HEADER] = CORS_ORIGIN_VALUE
    response.headers[CORS_METHOD_HEADER] = CORS_METHOD_VALUE
    return response


