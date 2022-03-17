from django.urls import path

from Visualizer.views import random_data

urlpatterns = [
    path('linear-regression/random-data', random_data, name = 'Linear Regression - Random Data')
]

