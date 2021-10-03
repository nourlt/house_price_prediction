import streamlit as st

import joblib
import pickle


import datetime
 

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder 
from sklearn.preprocessing import StandardScaler

st.sidebar.image("./hpp.jpeg")
tab = st.sidebar.columns(2)


today = datetime.date.today()

tab[0].date_input('Today',today)

tab[1].time_input("time")
st.sidebar.file_uploader('File uploader',accept_multiple_files=True,type=["pdf","jpg"])

#Separate categorical values and Numerical Values
Cat_Col = ['category','city','type']
Num_Col = ['room_count','bathroom_count' , 'size']

Pipeline = ColumnTransformer([
    ("num", StandardScaler(), Num_Col),
    ('cat', OrdinalEncoder(),Cat_Col)
])

df = pd.read_csv("./Immobiliers.csv")
Pipeline.fit_transform(df.drop(['price','log_price','region'],axis=1))
st.title("House Price Prediction")

room_numbers = st.number_input("put numbers of rooms",min_value=1)

bathroom_numbers = st.number_input("put numbers of bathrooms",min_value=1)

category = st.selectbox("Choose the category",['Terrains et Fermes', 'Appartements', 'Locations de vacances',
       'Magasins, Commerces et Locaux industriels', 'Maisons et Villas',
       'Colocations', 'Bureaux et Plateaux'])

surface = st.number_input("put size",min_value=30.0)

types = st.selectbox("put the caracteristics",['À Vendre', 'À Louer'])
city = st.selectbox("the Localisation:",['Ariana', 'Béja', 'Ben arous', 'Bizerte', 'Gabès', 'Gafsa',
       'Jendouba', 'Kairouan', 'Kasserine', 'Kébili', 'La manouba',
       'Le kef', 'Mahdia', 'Médenine', 'Monastir', 'Sidi bouzid',
       'Siliana', 'Sousse', 'Tataouine', 'Tozeur', 'Zaghouan', 'Sfax',
       'Nabeul', 'Tunis'])

c = st.columns(10)
pred = c[5].button("predict")

with open("./pipe.h5","rb") as f:
    pipe = pickle.load(f)

df_test = pd.DataFrame([[str(category),float(room_numbers),int(bathroom_numbers),float(surface),str(types),str(city)]],columns=["category","room_count","bathroom_count","size","type","city"])
df_test = Pipeline.transform(df_test)


model = joblib.load("./best.h5")
prediction = model.predict(df_test)
if pred:
       c = st.columns(10)
       st.balloons()
       c[5].success("{:.2f}".format(prediction[0]))
