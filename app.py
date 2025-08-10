import os
import gdown
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

# Google Drive fayl ID
FILE_ID = "1ChHrbdN-3w8d-sjI8gv16zvSn57tMJxa"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "fruit_model.h5"

# Modeli yükləmə funksiyası
def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(URL, MODEL_PATH, quiet=False)

# Modeli yüklə
download_model()
model = tf.keras.models.load_model(MODEL_PATH)

# Modelin input ölçüsünü avtomatik götür
input_shape = model.input_shape[1:3]  # (height, width)

def preprocess_image(image: Image.Image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = ImageOps.fit(image, input_shape, Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Streamlit UI
st.title("🍎 Meyvə keyfiyyəti təyini")

uploaded_file = st.file_uploader("JPG şəkil yüklə", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklədiyiniz şəkil', use_column_width=True)

    input_arr = preprocess_image(image)
    prediction = model.predict(input_arr)[0]

    rotten_prob = prediction[0]
    good_prob = prediction[1]

    if good_prob > rotten_prob:
        label = "Good (Sağlam)"
        confidence = good_prob
    else:
        label = "Rotten (Xarab)"
        confidence = rotten_prob

    st.markdown(f"**Nəticə:** {label}")
    st.markdown(f"**Əminlik faizi:** {confidence*100:.2f}%")
