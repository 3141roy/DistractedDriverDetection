import io
from PIL import Image
import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
import pandas as pd

model = load_model("myModel.h5")

MODEL_PATH = "myModel.h5"
LABELS_PATH = "custom_model/model_classes.txt"


def load_image():
    uploaded_file = st.file_uploader(label="Pick an image to test")
    if uploaded_file is not None:
        image_data = uploaded_file.getvalue()
        st.image(image_data)
        return Image.open(io.BytesIO(image_data))
    else:
        return None


# def load_model(model_path):
#     model = torch.load(model_path, map_location='cpu')
#     model.eval()
#     return model


def load_labels(labels_file):
    with open(labels_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
        return categories


def predict(model, image):
    image = image.resize((227, 227))
    #   imgplot = plt.imshow(image)
    #   plt.show()
    img = tf.keras.preprocessing.image.img_to_array(image)
    img = img / 255.0
    img = img.reshape(-1, 227, 227, 3)

    pred = model.predict(img)
    st.write(pred)
    attribute = [
        "safe driving",
        "texting - right",
        "talking on the phone - right",
        "texting - left",
        "talking on the phone - left",
        "operating the radio",
        "drinking",
        "reaching behind",
        "hair and makeup",
        "talking to passenger",
    ]
    index = np.argmax(pred[0])
    st.write(
        "The model when run on the given image returns the following classification : "
    )
    st.write(attribute[index])


def main():
    st.title("Custom model demo")
    # model = load_model(MODEL_PATH)
    # categories = load_labels(LABELS_PATH)
    image = load_image()
    result = st.button("Run on image")
    if result:
        st.write("Calculating results...")
        predict(model, image)


if __name__ == "__main__":
    main()
