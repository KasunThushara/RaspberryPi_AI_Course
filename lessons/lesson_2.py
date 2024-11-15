import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = "models/2.tflite"
LABEL_PATH = "models/imagenet-classes.txt"


@st.cache_resource
def load_efficientnet_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter


@st.cache_data
def load_labels():
    with open(LABEL_PATH, "r") as f:
        return [line.strip() for line in f.readlines()]


def preprocess_image_efficientnet(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = np.array(image).astype("uint8")
    image = np.expand_dims(image, axis=0)
    return image


def predict_image(interpreter, image):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


def lesson_2():
    st.markdown("""
        **This lesson is designed to test your trained model using ImageNet with up to 1000 classes.**
        You do not need to upload anything as we have prepared everything for you. 
        You only need to upload the image, and the model will classify it based on ImageNet's classes.
    """)
    st.markdown("""
        You can find more information about configuring TensorFlow on the Raspberry Pi environment in the following link:
        [Introduction to TensorFlow in Raspberry Pi Environment](https://seeed-projects.github.io/Tutorial-of-AI-Kit-with-Raspberry-Pi-From-Zero-to-Hero/docs/Chapter%202%20-%20Configuring%20the%20RaspberryPi%20Environment/Introduction_to_TensorFlow_in_Raspberry_Pi_Environment)
    """)

    st.subheader("EfficientNet Image Classification")
    interpreter = load_efficientnet_model()
    labels = load_labels()

    image_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if image_file:
        image = Image.open(image_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        input_image = preprocess_image_efficientnet(image)
        predictions = predict_image(interpreter, input_image)
        top_k = 3
        top_indices = np.argsort(predictions[0])[-top_k:][::-1]
        st.write("Top 3 Predictions:")
        for i in top_indices:
            st.write(f"{labels[i]}: Probability {predictions[0][i]:.4f}")
