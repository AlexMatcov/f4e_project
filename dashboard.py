import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

st.title("OPTION PRICE DASHBOARD")
st.write("by [Group 2](https://canvas.utwente.nl/groups/92289/users)")

st.sidebar.title("Parameters")

# sidebar inputs
S = st.sidebar.number_input("Stock price", min_value=0.0, value=100.0)
K = st.sidebar.number_input("Strike price", min_value=0.0, value=105.0)
v = st.sidebar.number_input("Volatility (%)", min_value=0.0, max_value=100.0, value=10.0)
v /= 100
T = st.sidebar.slider("Time horizon (years)", min_value=1)
r = st.sidebar.number_input("Risk-free rate (%)", min_value=0.0, max_value=100.0, value=5.0)
r /= 100
n = st.sidebar.number_input("Number of time steps", min_value=1, value=10)
m = st.sidebar.number_input("Number of simulation steps", min_value=1, value=100)
ep = st.sidebar.selectbox("Exercise policy", ("European", "American"))
cp = st.sidebar.selectbox("Option", ("Call", "Put"))

st.subheader("Introduction")
st.markdown("Option is a right buy or sell the stock at a certain price with maturity. Since buying a beneficial "
            "option will be a difficult work without any information about the option prices, the revenue or the loss "
            "should be shown for the investors to make the profit from it. To make the buyers trade about options "
            "easier, a product that does the complicated calculations could be a useful tool. The product for this "
            "project is a dashboard that calculates the option prices including exotic options. Monte-Carlo "
            "simulation is used as a formula for calculation. ")

st.header("Option Prices")

# general formulas
time_step = T / n
u = np.exp(v * np.sqrt(time_step))
d = 1 / u
p = (np.exp(r * time_step) - d) / (u - d)
q = 1 - p


def binomial_lattice(S, K, r, n, call_put, ep):
    stock_price = np.zeros((n + 1, n + 1))

    stock_price[0, 0] = S

    if ep == "European":
        for i in range(1, n + 1):
            stock_price[i, 0] = stock_price[i - 1, 0] * u
            for j in range(1, i + 1):
                stock_price[i, j] = stock_price[i - 1, j - 1] * d

    df_stock_price = pd.DataFrame(data=stock_price)
    df_stock_price = df_stock_price.T

    option_value = np.zeros((n + 1, n + 1))

    for i in range(n + 1):
        if call_put == 'Call':
            option_value[n, i] = max(0, stock_price[n, i] - K)
        elif call_put == 'Put':
            option_value[n, i] = max(0, K - stock_price[n, i])

    discount = np.exp(-r * time_step)

    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            option_value[i, j] = discount * (p * option_value[i + 1, j] + q * option_value[i + 1, j + 1])

    return option_value[0, 0], df_stock_price


binomial_price, df = binomial_lattice(S, K, r, n, cp, ep)

st.dataframe(df)
st.write(ep, cp, 'price: %.2f' % binomial_price)

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


def monte_carlo():
    drift = (r - (v ** 2) / 2) * time_step
    a = v * np.sqrt(time_step)
    x = np.random.normal(0, 1, (m, n))

    stock_price = np.zeros((m, n))
    stock_price[:, 0] += S

    for i in range(1, n):
        stock_price[:, i] += stock_price[:, i - 1] * np.exp(drift + a * x[:, i])

    call_array = stock_price[:, -1] - K
    for i in range(len(call_array)):
        if call_array[i] < 0:
            call_array[i] = 0
        else:
            call_array[i] = call_array[i]

    put_array = K - stock_price[:, -1]
    for i in range(len(put_array)):
        if put_array[i] < 0:
            put_array[i] = 0
        else:
            put_array[i] = put_array[i]

    payoff_call = np.mean(call_array)
    payoff_put = np.mean(put_array)

    call = payoff_call * np.exp(-r * T)
    put = payoff_put * np.exp(-r * T)

    if cp == "Call":
        return call, call_array
    else:
        return put, put_array


options_array = monte_carlo()[1]
no_zeros_option_array = np.delete(options_array, np.where(options_array == 0))

fig, ax = plt.subplots()
ax.hist(no_zeros_option_array, bins=15)
st.pyplot(fig)
st.write(cp, 'price: %.2f' % monte_carlo()[0])

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
