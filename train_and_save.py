import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

print("Loading data...")
with_mask_files = os.listdir('data/with_mask')
without_mask_files = os.listdir('data/without_mask')

with_mask_labels = [1]*len(with_mask_files)
without_mask_labels = [0]*len(without_mask_files)
labels = with_mask_labels + without_mask_labels

data = []
with_mask_path = 'data/with_mask/'
for file in with_mask_files:
    img = Image.open(with_mask_path + file).resize((128, 128)).convert('RGB')
    data.append(np.array(img))

without_mask_path = 'data/without_mask/'
for file in without_mask_files:
    img = Image.open(without_mask_path + file).resize((128, 128)).convert('RGB')
    data.append(np.array(img))

print("Data loaded. Converting to numpy arrays...")
X = np.array(data)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)
X_train_scaled = X_train/255
X_test_scaled = X_test/255

print("Building model...")
num_of_classes = 2
model = keras.Sequential([
    keras.layers.Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(128,128,3)),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Conv2D(64, kernel_size=(3,3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2,2)),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_of_classes, activation='sigmoid')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])

print("Training model...")
model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=5)

print("Evaluating...")
loss, accuracy = model.evaluate(X_test_scaled, y_test)
print('Test Accuracy =', accuracy)

print("Saving model...")
model.save('face_mask_model.h5')
print("Saved!")
