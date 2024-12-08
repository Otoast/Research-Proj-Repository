from django.urls import path
from . import views


urlpatterns = [
    path("", views.home, name='home'),
    path("submit", views.parse_RISK_submit, name='parse_risk_submit'),
    path("basic-calculator", views.basic_RISK_calculator, name='basic_calculator'),
    path("location-calculator", views.location_RISK_calculator, name='location_calculator')
]