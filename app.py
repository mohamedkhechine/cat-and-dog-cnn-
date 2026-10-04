import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('cat_dog_model.keras')

model = load_model()
st.title("Cat vs Dog Classifier")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Read the image file
    image = tf.image.decode_image(uploaded_file.read(), channels=3)
    image = tf.image.resize(image, [224, 224])
    image = preprocess_input(image)  # Preprocess the image
    image = tf.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    predictions = model.predict(image)
    class_names = ['Cat', 'Dog']
    predicted_class = class_names[tf.argmax(predictions[0])]

    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True) # new
    st.write(f"Predicted Class: {predicted_class}")