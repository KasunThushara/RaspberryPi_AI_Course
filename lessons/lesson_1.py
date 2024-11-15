import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf



class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def preprocess_image(image, target_size=(32, 32)):
    image = image.resize(target_size)
    image = np.array(image).astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def lesson_1():
    st.subheader("TFLite Model Image Classification")
    st.write("Upload a TFLite model and an image to perform predictions.")
    st.markdown("""
        You can create the model using the code in the following [Colab notebook](https://colab.research.google.com/github/Seeed-Projects/Tutorial-of-AI-Kit-with-Raspberry-Pi-From-Zero-to-Hero/blob/main/notebook/Chapter1/TensorFlow_CNN.ipynb):

        <a target="_blank" href="https://colab.research.google.com/github/Seeed-Projects/Tutorial-of-AI-Kit-with-Raspberry-Pi-From-Zero-to-Hero/blob/main/notebook/Chapter1/TensorFlow_CNN.ipynb">
          <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
        </a>
    """, unsafe_allow_html=True)

    model_file = st.file_uploader("Upload TFLite model", type=["tflite"])
    if model_file:
        model_path = "loaded.tflite"
        with open(model_path, "wb") as f:
            f.write(model_file.read())
        interpreter = load_tflite_model(model_path)
        st.success("Model loaded successfully!")

        image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if image_file:
            image = Image.open(image_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)  # Updated parameter
            input_image = preprocess_image(image)
            predictions = predict_image(interpreter, input_image)
            predicted_label = np.argmax(predictions)
            st.write(f"Predicted label: {class_names[predicted_label]} ({predicted_label})")