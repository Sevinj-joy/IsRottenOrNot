import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Modeli yüklə
model = tf.keras.models.load_model('fruit_model.h5')


# Funksiya şəkili preprocess etmək üçün (modelin gözlədiyi input ölçüsünə görə dəyişə bilər)
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    # RGB-ə çevir
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Şəkili resize et
    image = ImageOps.fit(image, target_size, Image.ANTIALIAS)
    # Numpy arrayə çevir və normallaşdır
    img_array = np.array(image) / 255.0
    # Batch dimension əlavə et
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Streamlit app
st.title("Fruit Quality Detection")
st.write("Yüklədiyiniz şəkilə əsasən meyvənin sağlam və ya xarab olduğunu müəyyən edir.")

uploaded_file = st.file_uploader("JPG şəkil yüklə", type=["jpg", "jpeg"])

if uploaded_file is not None:
    # Şəkili aç
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklədiyiniz şəkil', use_column_width=True)

    # Şəkili hazırlamaq
    input_arr = preprocess_image(image, target_size=(224, 224))  # ölçünü modelə uyğun dəyiş

    # Proqnoz ver
    prediction = model.predict(input_arr)[0]  # [0] - batch üçün
    # Tutaq ki, model 2 sinifli (0 = Rotten, 1 = Good)

    rotten_prob = prediction[0]
    good_prob = prediction[1]

    if good_prob > rotten_prob:
        label = "Good (Sağlam)"
        confidence = good_prob
    else:
        label = "Rotten (Xarab)"
        confidence = rotten_prob

    st.markdown(f"**Nəticə:** {label}")
    st.markdown(f"**Əminlik faizi:** {confidence * 100:.2f}%")
