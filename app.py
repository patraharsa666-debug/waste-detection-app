import streamlit as st
from ultralytics import YOLO
from PIL import Image

st.title("Waste Detection using YOLOv8")

model = YOLO("last.pt")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    results = model.predict(image)

    annotated_image = results[0].plot()

    st.image(
        annotated_image,
        caption="Detection Result",
        use_container_width=True
    )

    boxes = results[0].boxes

    st.write(f"Objects detected: {len(boxes)}")
