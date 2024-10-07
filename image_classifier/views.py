from keras.models import load_model
import os
from django.conf import settings
model = load_model(os.path.join(settings.BASE_DIR, 'classifier', 'model.h5'))
