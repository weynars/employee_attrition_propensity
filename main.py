import streamlit as st
import pandas as pd
import numpy as np

# import plotly.express as px
import plotly.figure_factory as ff

st.title('My first app')

filename = './data/healthcare_data.csv'

@st.cache_data
def fetch_data(filename):
    df = pd.read_csv(filename)
    return df

df = fetch_data(filename)

col = 'Age'

min_value = int(df[col].min())
max_value = int(df[col].max())

with st.container():
    st.write('**Filter**')
    values = st.slider(col, min_value=min_value, max_value=max_value, value=(25,max_value), step=1)

# fig = px.scatter(df, x = "DailyRate", y = "DistanceFromHome", color = "Attrition", title = title)
fig = ff.create_distplot([df[col]], [col], bin_size=5, show_rug=False)
fig.add_vrect(x0=values[0], x1=values[1], line_width=0, fillcolor="red", opacity=0.2)

st.plotly_chart(fig, theme="streamlit", use_container_width=True)

st.table(df.head())



