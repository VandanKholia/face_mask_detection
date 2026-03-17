from tensorflow.keras.models import load_model

model = load_model("face_mask_model.h5")
model.save("face_mask_model.keras")

print("Model converted successfully!")