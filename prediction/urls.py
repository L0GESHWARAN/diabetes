from django.urls import path
from . import views

urlpatterns=[
    path('',views.home, name='home'),
    path('main/',views.main, name='index'),
    path('main/result',views.result,name='result'),
    path('main/result/more',views.result,name='more'),
]