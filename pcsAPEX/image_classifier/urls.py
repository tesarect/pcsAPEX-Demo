from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('images/', views.ImageListView.as_view(), name='image_list'),
    path('images/<int:pk>/', views.ImageDetailView.as_view(), name='image_detail'),
    path('images/upload/', views.upload_image, name='upload_image'),
    path('images/<int:pk>/label/', views.label_image, name='label_image'),
    path('images/<int:pk>/edit-label/', views.edit_image_label, name='edit_image_label'),
    path('images/<int:pk>/confirm-label/', views.confirm_image_label, name='confirm_image_label'),
    path('categories/', views.CategoryListView.as_view(), name='category_list'),
    path('categories/create/', views.CategoryCreateView.as_view(), name='category_create'),
    path('training-jobs/', views.TrainingJobListView.as_view(), name='training_job_list'),
    path('training-jobs/create/', views.TrainingJobCreateView.as_view(), name='training_job_create'),
    path('training-jobs/<int:pk>/', views.TrainingJobDetailView.as_view(), name='training_job_detail'),
    path('training-jobs/<int:pk>/start/', views.start_training, name='start_training'),
    path('process-new-images/', views.process_new_images, name='process_new_images'),
    path('system-config/', views.system_config, name='system_config'),
]
