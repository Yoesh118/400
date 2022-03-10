from django.urls import path

from . import views

urlpatterns = [
    path('api/v1/process/<int:id>/', views.EntryPoint.as_view(), name='entry'),
]


