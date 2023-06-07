import tensorflow as tf
from PIL import Image

# Load the image

image = Image.open('./test/NORMAL/IM-0001-0001.jpeg')

# Preprocess the image

image = image.resize((224, 224)) # Resize the image
image = image.convert('RGB')
image_array = tf.keras.preprocessing.image.img_to_array(image) # Convert the image to an array
preprocessed_image = tf.keras.applications.resnet50.preprocess_input(image_array) # Apply model-specific preprocessing

# Load the model

model = tf.keras.models.load_model("model_pneumonia.h5")

# Expand the dimensions of the image array to match the model's input shape

expanded_image = tf.expand_dims(preprocessed_image, axis=0)

#Make predictions with the image

predictions = model.predict(expanded_image)
print(predictions)