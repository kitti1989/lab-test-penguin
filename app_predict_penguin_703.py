import streamlit as st

import pickle
import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

model, species_encoder, island_encoder ,sex_encoder = pickle.load(open('knn_penguin_703.pkl', 'rb'))

# ชื่อเพจของแอป Streamlit
st.title("Penguin Data Input")

# สร้างฟอร์มสำหรับกรอกข้อมูล
st.subheader("Input Penguin Data")

# ค่าตั้งต้นในฟอร์ม
default_island = 'Torgersen'
default_culmen_length_mm = 37.0
default_culmen_depth_mm = 19.3
default_flipper_length_mm = 192.3
default_body_mass_g = 3750
default_sex = 'MALE'

# สร้างอินพุตในฟอร์ม
island = st.selectbox("Island", ['Torgersen', 'Biscoe', 'Dream'], index=0)
culmen_length_mm = st.number_input("Culmen Length (mm)", value=default_culmen_length_mm)
culmen_depth_mm = st.number_input("Culmen Depth (mm)", value=default_culmen_depth_mm)
flipper_length_mm = st.number_input("Flipper Length (mm)", value=default_flipper_length_mm)
body_mass_g = st.number_input("Body Mass (g)", value=default_body_mass_g)
sex = st.selectbox("Sex", ['MALE', 'FEMALE'], index=0)

# สร้างปุ่มสำหรับการแสดงข้อมูลที่กรอก
if st.button("Predict"):
    # เตรียมข้อมูลสำหรับทำนาย
    x_new = pd.DataFrame({
        'island': [island_encoder.transform([island])[0]],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex_encoder.transform([sex])[0]]
    })

    # ทำนาย species โดยใช้โมเดลที่โหลดมา
    prediction = model.predict(x_new)
    species = species_encoder.inverse_transform(prediction)[0]

    # แสดงผลลัพธ์ species
    st.subheader("Predicted Species")
    st.write(f"The predicted species is: **{species}**")
