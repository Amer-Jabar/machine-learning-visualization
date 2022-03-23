from django.urls import path

from Visualizer.linear_regression_views import (
    x_discrete_steps,
    base_endpoint as linear_base_endpoint,
    random_data as linear_random_data,
    generate_coeffs as linear_generate_coeffs,
    execute_algo as linear_execute_algo
)
from Visualizer.logistic_regression_views import (
    base_endpoint as logistic_base_endpoint,
    random_data as logistic_random_data,
    generate_coeffs as logistic_generate_coeffs,
    execute_algo as logistic_execute_algo
)

urlpatterns = [
    path('linear-regression/', linear_base_endpoint, name = 'Linear Regression'),
    path('linear-regression/random-data', linear_random_data, name = 'Linear Regression - Random Data'),
    path('linear-regression/x-steps', x_discrete_steps, name = 'Linear Regression - X Steps'),
    path('linear-regression/coeffs', linear_generate_coeffs, name = 'Linear Regression - Generate Coeffs'),
    path('linear-regression/execute', linear_execute_algo, name = 'Linear Regression - Execute Algorithm'),
    path('logistic-regression/', logistic_base_endpoint, name = 'Logistic Regression'),
    path('logistic-regression/random-data', logistic_random_data, name = 'Logistic Regression - Random Data'),
    path('logistic-regression/coeffs', logistic_generate_coeffs, name = 'Logistic Regression - Generate Coeffs'),
    path('logistic-regression/execute', logistic_execute_algo, name = 'Logistic Regression - Execute Algorithm'),
]

