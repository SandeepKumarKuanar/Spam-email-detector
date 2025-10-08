from django.apps import AppConfig
from django.conf import settings # Import Django's settings
import joblib
import os

class ApiConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'api'

    # --- Use BASE_DIR for reliable paths ---
    MODEL_PATH = os.path.join(settings.BASE_DIR, 'model', 'spam_detector_model.joblib')
    VECTORIZER_PATH = os.path.join(settings.BASE_DIR, 'model', 'tfidf_vectorizer.joblib')

    # Load model and vectorizer
    try:
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        print("✅ Model and vectorizer loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model or vectorizer: {e}")
        model = None
        vectorizer = None