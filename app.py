import os
import gdown
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import streamlit as st

FILE_ID = "1ChHrbdN-3w8d-sjI8gv16zvSn57tMJxa"
URL = f"https://drive.google.com/uc?id={FILE_ID}"
MODEL_PATH = "fruit_model.h5"

def download_model():
    if not os.path.exists(MODEL_PATH):  # təkrar yükləməsin deyə
        gdown.download(URL, MODEL_PATH, quiet=False)

download_model()

model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image: Image.Image, target_size=(224, 224)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Yeni Pillow versiyası üçün dəyişiklik
    image = ImageOps.fit(image, target_size, Image.Resampling.LANCZOS)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

st.title("🍎 Meyvə Keyfiyyəti Təyini")

uploaded_file = st.file_uploader("📤 JPG şəkil yüklə", type=["jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklədiyiniz şəkil', use_column_width=True)

    input_arr = preprocess_image(image, target_size=(224, 224))
    prediction = model.predict(input_arr)[0]

    rotten_prob = prediction[0]
    good_prob = prediction[1]

    if good_prob > rotten_prob:
        label = "✅ Sağlam"
        confidence = good_prob
    else:
        label = "❌ Xarab"
        confidence = rotten_prob

    st.markdown(f"**Nəticə:** {label}")
    st.markdown(f"**Əminlik faizi:** {confidence*100:.2f}%")
