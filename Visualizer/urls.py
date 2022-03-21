from django.urls import path

from Visualizer.views import base_endpoint, random_data, x_discrete_steps, generate_coeffs, execute_algo

urlpatterns = [
    path('linear-regression/', base_endpoint, name = 'Linear Regression'),
    path('linear-regression/random-data', random_data, name = 'Linear Regression - Random Data'),
    path('linear-regression/x-steps', x_discrete_steps, name = 'Linear Regression - X Steps'),
    path('linear-regression/coeffs', generate_coeffs, name = 'Linear Regression - Generate Coeffs'),
    path('linear-regression/execute', execute_algo, name = 'Linear Regression - Execute Algorithm'),
]

