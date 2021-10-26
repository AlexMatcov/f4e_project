import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("OPTION PRICE DASHBOARD")
st.write("by [Group 2](https://canvas.utwente.nl/groups/92289/users)")

st.sidebar.title("Parameters")

# sidebar inputs
S = st.sidebar.number_input("Stock price", min_value=0.0, value=100.0)
K = st.sidebar.number_input("Strike price", min_value=0.0, value=105.0)
T = st.sidebar.slider("Time horizon (years)", min_value=1)
r = st.sidebar.number_input("Risk-free rate (%)", min_value=0.0, max_value=100.0, value=5.0)
n = st.sidebar.slider("Number of time steps", min_value=1, value=20)
ep = st.sidebar.selectbox("Exercise policy", ("American", "European"))

st.header("Option Prices")
st.header("Monte-Carlo Simulation")