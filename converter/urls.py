"""
URL configuration for converter app.
"""
from django.urls import path
from . import views

app_name = 'converter'

urlpatterns = [
    path('', views.UploadView.as_view(), name='upload'),
    path('process/', views.ProcessView.as_view(), name='process'),
    path('download/<str:filename>/', views.DownloadView.as_view(), name='download'),
    path('cleanup/', views.CleanupView.as_view(), name='cleanup'),
]