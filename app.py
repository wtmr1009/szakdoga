import streamlit as st
import pandas as pd

st.title("MCDA Patient Pathway Finder")

csv_file = st.file_uploader("Upload CSV", type="csv")
if csv_file:
    df = pd.read_csv(csv_file)
    st.write("Intézmények szolgáltatásai, árai és helyszínei:")
    st.dataframe(df[['institution', 'service', 'price', 'location']])
