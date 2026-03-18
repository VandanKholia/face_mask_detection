"""
Run this script once to convert face_mask_model.keras -> face_mask_model.h5
Then push the .h5 file and update app.py to load it.
"""
import tensorflow as tf
from tensorflow.keras.models import load_model

print("Loading .keras model...")
model = load_model('backend/face_mask_model.keras', compile=False)

print("Saving as .h5...")
model.save('backend/face_mask_model.h5')

print("Done! face_mask_model.h5 saved in backend/")
