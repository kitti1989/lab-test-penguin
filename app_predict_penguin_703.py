import streamlit as st

import pickle
import numpy as np
import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

model, species_encoder, island_encoder ,sex_encoder = pickle.load(open('knn_penguin_703.pkl', 'rb'))

st.title("Penguin Species prediction using KNN")

st.write("## Input Penguin Information")
island = st.selectbox('Island', ('Torgersen', 'Biscoe', 'Dream'))
culmen_length_mm = st.slider('culmen length mm', 0.0, 100.0, 37.0)
culmen_depth_mm = st.slider('culmen depth per mm', 0.0, 100.0, 19.3)
flipper_length_mm = st.slider('flipper length per mm', 0.0, 1000.0, 192.3)
body_mass_g = st.slider('body mass per g', 0.0, 10000.0, 3750.0)
sex = st.selectbox('Sex', ('MALE', 'FEMALE'))

x_new = pd.DataFrame() 
x_new['culmen_length_mm'] = [culmen_length_mm]
x_new['culmen_depth_mm'] = [culmen_depth_mm]
x_new['flipper_length_mm'] = [flipper_length_mm]
x_new['body_mass_g'] = [body_mass_g]
x_new['island'] = island_encoder.transform(x_new['island'])
x_new['sex'] = sex_encoder.transform(x_new['sex'])

pred = model.predict(x_new)

st.write("## Prediction Result:")
st.write("Species: ", pred[0])
