from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='login'),
    path('validate/', views.send_otp, name='send_otp'),
    path('validate/otp/', views.validate_otp, name='validate_otp'),
    path('validate/otp/resend', views.resent_otp, name='resend_otp'),
    path('validate/otp/feedback/', views.api_page, name='home'),
    path('validate/otp/feedback/success/', views.success, name='success'),
    path('validate/otp/feedback/success/results',
         views.results, name='results'),
    path('validate/otp/feedback/success/model', views.model, name='model')
]
