from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('download/', views.download_file, name='download_file'),
    path('download_report/', views.download_report, name='download_report'),
    path('download_basic_stats/', views.download_basic_stats, name='download_basic_stats'),
    path('download_selected_features/', views.download_selected_features, name='download_selected_features'),
    path('download_pca/', views.download_pca, name='download_pca'),
]