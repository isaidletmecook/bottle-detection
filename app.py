import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.set_page_config(page_title="Bottle Detection", layout="centered")

st.title("Bottle Detection using YOLOv8")
st.write("Upload an image and the model will detect bottles.")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Detecting bottles..."):
        results = model.predict(image)

    result_image = results[0].plot()
    st.image(result_image, caption="Detection Result", use_column_width=True)
