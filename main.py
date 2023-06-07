from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Set the image size
img_height, img_width = 224, 224

# Create an ImageDataGenerator object to normalize the images and apply augmentation transformations
train_datagen = ImageDataGenerator(
    rescale=1./255,  # normalize pixel values between 0 and 1
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Load the images using flow_from_directory
train_generator = train_datagen.flow_from_directory(
    directory="./train/",
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary')

# Model evaluation
# Create an ImageDataGenerator object for normalizing the images
eval_datagen = ImageDataGenerator(rescale=1./255)

# Load the images using flow_from_directory
eval_generator = eval_datagen.flow_from_directory(
    directory="./val/",
    target_size=(img_height, img_width),
    batch_size=32,
    class_mode='binary')

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model on the training data
model.fit(train_generator, epochs=3)

# Get class indices
class_indices = train_generator.class_indices

# Evaluate the model on the evaluation data
loss, accuracy = model.evaluate(eval_generator)

# Save the model
model.save('model_pneumonia.h5')
