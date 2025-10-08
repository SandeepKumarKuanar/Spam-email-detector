from django.urls import path
from .views import PredictView

urlpatterns = [
    # This line matches 'predict/' and connects it to your PredictView
    path('predict/', PredictView.as_view(), name='predict'),
]