import streamlit as st
import cv2
import numpy as np
import time
from ultralytics import YOLO

# Load COCO class names
with open("models/coco.txt", "r") as f:
    class_names = f.read().splitlines()

# Load the YOLOv11 TFLite model
model = YOLO("models/best_float16.tflite")


# Function to detect objects and annotate the frame
def detect_objects(frame):
    results = model(frame, conf=0.7, imgsz=224)

    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get coordinates
            confidence = box.conf[0]  # Confidence score
            class_id = int(box.cls[0])  # Class ID
            class_name = class_names[class_id]  # Class name

            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f'{class_name} {confidence:.2f}'
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame


# Display the stream in Streamlit
def display():
    st.subheader("Continuous Webcam Object Detection")
    st.write("This application continuously captures frames from your webcam and runs YOLOv8 detection.")

    # Open the webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open the webcam.")
        return

    # Initialize an empty placeholder for the video stream
    frame_placeholder = st.empty()

    # Continuous stream loop
    #prev_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame from webcam.")
            break

        # Detect objects
        frame = detect_objects(frame)

        # Convert the frame to RGB (OpenCV uses BGR by default)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the single frame placeholder with the latest frame
        frame_placeholder.image(frame, channels="RGB", use_container_width=True)

        # Calculate FPS and display it in the Streamlit sidebar
        #curr_time = time.time()
        #fps = 1 / (curr_time - prev_time)
        #prev_time = curr_time
        #st.sidebar.write(f"**FPS:** {fps:.2f}")

        # Small delay to simulate real-time processing
        time.sleep(0.05)

    # Release the webcam
    cap.release()
