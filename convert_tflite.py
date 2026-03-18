import tensorflow as tf

print("Loading .h5 model...")
model = tf.keras.models.load_model('backend/face_mask_model.h5', compile=False)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

print("Saving TFLite model...")
with open('backend/model.tflite', 'wb') as f:
    f.write(tflite_model)

print("Successfully converted to backend/model.tflite!")
