#main.py
import streamlit as st
import pandas as pd
from numpy.random import default_rng as rng

df = pd.DataFrame(
    rng(0).standard_normal((50,20)), columns=([f"col {d}" for d in range(20)])
)

st.dataframe(df)
