import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
from PIL import Image

# Load the trained YOLOv8 model
model = YOLO(r"C:\Users\pawan\Surveillance-System\model\best (1).pt")  # Replace with the path to your trained model

# Streamlit UI
st.set_page_config(page_title="Survelliance System", page_icon="🔍", layout="wide")
st.title("🔍 Survelliance System")
st.write("Upload an image or video for object detection using the powerful model.")

# Sidebar for input options
st.sidebar.header("Choose Input Type")
mode = st.sidebar.selectbox("Select input type:", ["Image", "Video"])

# Display a banner or logo (optional)
st.image(r"C:\Users\pawan\Surveillance-System\Image\cctv-security-surveillance-camera-icon-vector-47656026.jpg", width=200)  # Replace with your logo path

# Image mode
if mode == "Image":
    st.subheader("Image Upload")
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Processing...")
        image_np = np.array(image)
        results = model.predict(image_np)
        result_image = results[0].plot()
        st.image(result_image, caption="Detected Image", use_column_width=True)

# Video mode
elif mode == "Video":
    st.subheader("Video Upload")
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.video(temp_video_path, start_time=0)
        st.write("Processing video, please wait...")

        # Process video
        cap = cv2.VideoCapture(temp_video_path)
        if cap.isOpened():
            stframe = st.empty()
            scale_factor = 0.5  # Adjust the scale to optimize processing speed
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                # Resize the frame for faster processing (optional)
                frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor)

                # Perform object detection
                results = model.predict(frame)
                result_frame = results[0].plot()

                # Display the detection frame in the Streamlit app
                stframe.image(result_frame, channels="BGR", use_column_width=True)

            cap.release()

# Footer or additional information
st.sidebar.markdown("---")
st.sidebar.write("Powered by YOLOv8 and Streamlit")
st.sidebar.write("Developed by [Pavan Kumar]")

