from rest_framework.views import APIView
from rest_framework.response import Response
from .apps import ApiConfig # Import the app config to access the model

# This is the class that your urls.py is looking for
class PredictView(APIView):
    def post(self, request):
        # Get the message from the POST request data
        message = request.data.get('message', '')

        if not message:
            return Response({'error': 'Message not provided'}, status=400)

        # Use the loaded model and vectorizer to make a prediction
        message_features = ApiConfig.vectorizer.transform([message])
        prediction = ApiConfig.model.predict(message_features)

        result = 'spam' if prediction[0] == 1 else 'ham'

        # Return the prediction in the response
        return Response({'prediction': result})