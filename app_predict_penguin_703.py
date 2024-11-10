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
    # เก็บข้อมูลที่กรอกลงใน DataFrame
    x_new = pd.DataFrame({
        'island': [island],
        'culmen_length_mm': [culmen_length_mm],
        'culmen_depth_mm': [culmen_depth_mm],
        'flipper_length_mm': [flipper_length_mm],
        'body_mass_g': [body_mass_g],
        'sex': [sex]
    })

    # แสดงข้อมูลที่กรอกใน DataFrame
    st.subheader("Input Data")
    st.write(x_new)
    
    # ทดแทนในส่วนนี้ด้วยการทำนายหรือประมวลผลข้อมูลของคุณ
    st.write("This is where prediction or further processing would happen.")
