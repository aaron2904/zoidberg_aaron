#!/usr/bin/env python

coding: utf-8

import tensorflow as tf
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

#Set seed for reproducibility

tf.random.set_seed(42)

#Paths

test_normal_path = "test/NORMAL/.jpeg"
test_pneumonia_path = "test/PNEUMONIA/.jpeg"

train_normal_path = "train/NORMAL/.jpeg"
train_pneumonia_path = "train/PNEUMONIA/.jpeg"

val_normal_path = "val/NORMAL/.jpeg"
val_pneumonia_path = "val/PNEUMONIA/.jpeg"

last_test_normal_path = "last_test/*.jpeg"

#Load test image paths

test_normal_files = glob.glob(test_normal_path)
test_pneumonia_files = glob.glob(test_pneumonia_path)

#Load train image paths

train_normal_files = glob.glob(train_normal_path)
train_pneumonia_files = glob.glob(train_pneumonia_path)

#Load validation image paths

val_normal_files = glob.glob(val_normal_path)
val_pneumonia_files = glob.glob(val_pneumonia_path)

#Load last test image paths

last_test_files = glob.glob(last_test_normal_path)

print("Number of normal test images:", len(test_normal_files))
print("Number of pneumonia test images:", len(test_pneumonia_files))
print("Number of normal train images:", len(train_normal_files))
print("Number of pneumonia train images:", len(train_pneumonia_files))

#Create matrix and label arrays

#Train

train_matrix = []
train_label = []

#Test

test_matrix = []
test_label = []

#Validation

val_matrix = []
val_label = []

last_test_matrix = []
last_test_label = []

def image_to_matrix(files, label, matrix_array, label_array):
for file in files:
file = tf.io.read_file(file)

# Decode the image into a pixel matrix
matrix = tf.image.decode_jpeg(file, channels=3)


# Resize the matrix
matrix = tf.image.resize(matrix, [150, 150])

# Normalize the pixels between 0 and 1
matrix = matrix / 255.0

# Append the matrix to the array
matrix_array.append(matrix)

# Append the label to the array
# 0 = Normal
# 1 = Pneumonia

label_array.append(label)
image_to_matrix(train_normal_files, 0, train_matrix, train_label)
image_to_matrix(train_pneumonia_files, 1, train_matrix, train_label)

image_to_matrix(test_normal_files, 0, test_matrix, test_label)
image_to_matrix(test_pneumonia_files, 1, test_matrix, test_label)

image_to_matrix(val_normal_files, 0, val_matrix, val_label)
image_to_matrix(val_pneumonia_files, 1, val_matrix, val_label)

image_to_matrix(last_test_files, 0, last_test_matrix, last_test_label)

print("Number of images in train matrix:", len(train_matrix))
print("Number of labels in train matrix:", len(train_label))

print("Number of images in test matrix:", len(test_matrix))
print("Number of labels in test matrix:", len(test_label))

print("Number of images in val matrix:", len(val_matrix))
print("Number of labels in val matrix:", len(val_label))

#Create the model

model = tf.keras.models.Sequential([
tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
tf.keras.layers.MaxPooling2D((2, 2)),
tf.keras.layers.Flatten(),
tf.keras.layers.Dense(512, activation='relu'),
tf.keras.layers.Dense(1, activation='sigmoid')
])

#Model summary

model.summary()

#Compile the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

#Convert arrays to tensors

train_matrix_np = np.array(train_matrix, dtype=np.float32)
train_label_np = np.array(train_label, dtype=np.float32)
test_matrix_np = np.array(test_matrix, dtype=np.float32)
test_label_np = np.array(test_label, dtype=np.float32)
val_matrix_np = np.array(val_matrix, dtype=np.float32)
val_label_np = np.array(val_label, dtype=np.float32)
last_test_matrix_np = np.array(last_test_matrix, dtype=np.float32)

#Train the model

history = model.fit(train_matrix_np, train_label_np, epochs=5, validation_data=(test_matrix_np, test_label_np))

#Plot accuracy

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

#Evaluate the model

test_loss, test_acc = model.evaluate(test_matrix_np, test_label_np, verbose=2)

print("Test loss:", test_loss)
print("Test accuracy:", test_acc)

#Save the model

model.save('model.h5')