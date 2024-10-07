from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
from keras.models import load_model
from keras.utils import img_to_array
import numpy as np
import os
from PIL import Image

# Load the model
model = load_model(os.path.join(settings.BASE_DIR, 'classifier', 'model.h5'))

# Image size that the model expects
IMAGE_SIZE = (180, 180)

@csrf_exempt
def upload_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if not image_file:
            return render(request, 'upload.html', {'error': 'No image provided'})

        # Save the uploaded image to the media directory
        file_name = default_storage.save(f"uploads/{image_file.name}", image_file)
        file_path = os.path.join(settings.MEDIA_ROOT, file_name)

        # Convert the uploaded image to a format that Keras can process
        try:
            image = Image.open(file_path)  # Open the image from the file path
            img = image.resize(IMAGE_SIZE)  # Resize the image to the required size
            img_array = img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Create batch axis

            # Make prediction
            predictions = model.predict(img_array)
            score = float(predictions[0][0])

            # Interpret the score
            if score >= 0.5:
                result = {
                    'label': 'Dog',
                    'confidence': f"{10 * score:.2f}%"
                }
            else:
                result = {
                    'label': 'Cat',
                    'confidence': f"{10 * (1 - score):.2f}%"
                }

            # Return the result and the image path to the template
            return render(request, 'upload.html', {
                'result': result,
                'image_url': file_name
            })
        except Exception as e:
            return render(request, 'upload.html', {'error': f"An error occurred: {str(e)}"})

    return render(request, 'upload.html')

