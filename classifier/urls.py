from django.urls import path
from .views import predict_image, upload_image

urlpatterns = [
    path('predict/', predict_image, name='predict_image'),
    path('upload/', upload_image, name='upload_image'),  # This route serves the form
]


