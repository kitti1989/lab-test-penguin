import streamlit as st

import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

model = pickle.load(open('knn_penguin_703.pkl', 'rb'))

st.title("Penguin Species prediction using KNN")

st.write("## Input Penguin Information")
island = st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
culmen_length_mm = st.slider('culmen length mm', 0.0, 100.0, 0.0)
culmen_depth_mm = st.slider('culmen depth per mm', 0.0, 100.0, 0.0)
flipper_length_mm = st.slider('flipper length per mm', 0.0, 100.0, 0.0)
body_mass_g = st.slider('body mass per g', 0.0, 100.0, 0.0)
sex = st.selectbox('Sex', ('MALE', 'FEMALE'))

xnew = np.array([[island, culmen_length_mm, culmen_depth_mm, flipper_length_mm, body_mass_g, sex,]])

pred = model.predict(xnew)

st.write("## Prediction Result:")
st.write("Species: ", pred[0])
