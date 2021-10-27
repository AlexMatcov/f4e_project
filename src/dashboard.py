import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

st.title("OPTION PRICE DASHBOARD")
st.write("by [Group 2](https://canvas.utwente.nl/groups/92289/users)")

st.sidebar.title("Parameters")

# sidebar inputs
S = st.sidebar.number_input("Stock price", min_value=0.0, value=100.0)
K = st.sidebar.number_input("Strike price", min_value=0.0, value=105.0)
v = st.sidebar.number_input("Volatility", min_value=0.0, max_value=1.0, value=0.1)
T = st.sidebar.slider("Time horizon (years)", min_value=1)
r = st.sidebar.number_input("Risk-free rate ", min_value=0.0, max_value=1.0, value=0.05)
n = st.sidebar.slider("Number of time steps", min_value=1, value=10)
ep = st.sidebar.selectbox("Exercise policy", ("American", "European"))
cp = st.sidebar.selectbox("Option", ("Call", "Put"))

st.subheader("Introduction")
st.markdown("Option is a right buy or sell the stock at a certain price with maturity. Since buying a beneficial "
            "option will be a difficult work without any information about the option prices, the revenue or the loss "
            "should be shown for the investors to make the profit from it. To make the buyers trade about options "
            "easier, a product that does the complicated calculations could be a useful tool. The product for this "
            "project is a dashboard that calculates the option prices including exotic options. Monte-Carlo "
            "simulation is used as a formula for calculation. ")

st.header("Option Prices")

def binomial_lattice(S, K, r, v, T, n, call_put, ep):
    time_step = T / n

    # Compute u and d
    u = np.exp(v * np.sqrt(time_step))
    d = 1 / u

    # Compute p and q
    p = (np.exp(r * time_step) - d) / (u - d)
    q = 1 - p

    # Create empty matrix for stock prices
    stock_price = np.zeros((n + 1, n + 1))

    # Set initial stock price
    stock_price[0, 0] = S

    # Fill matrix with stock prices per time step
    if ep == "European":
        for i in range(1, n + 1):
            stock_price[i, 0] = stock_price[i - 1, 0] * u
            for j in range(1, i + 1):
                stock_price[i, j] = stock_price[i - 1, j - 1] * d

    # Transform numpy matrix into Pandas Dataframe
    df_stock_price = pd.DataFrame(data=stock_price)
    df_stock_price = df_stock_price.T

    # Create empty matrix for option values
    option_value = np.zeros((n + 1, n + 1))

    # For final time step, compute option value based on stock price and strike price
    for i in range(n + 1):
        if call_put == 'Call':
            option_value[n, i] = max(0, stock_price[n, i] - K)
        elif call_put == 'Put':
            option_value[n, i] = max(0, K - stock_price[n, i])

    # Compute discount factor per time step
    discount = np.exp(-r * time_step)

    # Recursively compute option value at time 0
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_value[i, j] = discount * (p * option_value[i + 1, j] + q * option_value[i + 1, j + 1])

    return option_value[0, 0], df_stock_price


# Test case: the following settings should yield an option price of 4.04

binomial_price, df = binomial_lattice(S, K, r, v, T, n, cp, ep)

df
st.write(cp, 'price: %.2f' % binomial_price)

st.subheader("Formulae Used")
st.latex(r"d = \frac{1}{u}")
st.latex(r"S_{t+1} = S_t \cdot u")
st.latex(r"S_{t+1} = S_t \cdot d")
st.latex("C_T = max(S_T-X, 0)")
st.latex("P_T = max(X-S_T, 0)")
st.latex(r"p = \frac{e^r - d}{u - d}")
st.latex(r"C_t = e^{-r} (p \cdot Cu_{t+1} + (1 - p) \cdot Cd_{t+1})")
st.latex(r"P_t = e^{-r} (p \cdot Pu_{t+1} + (1 - p) \cdot Pd_{t+1})")
st.write("\n")

st.header("Monte-Carlo Simulation")

image = Image.open("images\monte_carlo.png")
st.image(image)
st.write("Output: 0.0")

st.subheader("What's Monte-Carlo Simulation?")
st.markdown(
    "The calculation of option prices is an incredibly hard task, and you can never know the outcome for sure. One of "
    "the most relevant parts of the calculations of option prices is the variance. Calculating the variance as "
    "precise as possible is therefore of great importance to have a relevant outcome. This is where a Monte-Carlo "
    "simulation comes in handy. A Monte-Carlo simulation can execute calculations multiple times, with different "
    "initial variables. By doing this, the variance might be more accurate than using another method, because the "
    "simulation processed many different input variables. ")

st.subheader("Formulae Used")
st.latex(r"\frac{f(x)}{g(x)} \approx k")
st.latex("G(x) = \int_0^x g(x)dx")