import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp, sqrt, log
from scipy.stats import norm
from matplotlib.backends.backend_agg import RendererAgg
_lock = RendererAgg.lock

st.title("OPTION PRICE DASHBOARD")
st.write("by [Group 2](https://canvas.utwente.nl/groups/92289/users)")

st.sidebar.title("Parameters")

# sidebar inputs
S = st.sidebar.number_input("Stock price", min_value=0.1, value=100.0)
K = st.sidebar.number_input("Strike price", min_value=0.1, value=105.0)
v = st.sidebar.number_input("Volatility (%)", min_value=0.1, max_value=100.0, value=10.0)
v /= 100
T = st.sidebar.slider("Time horizon (years)", min_value=1)
r = st.sidebar.number_input("Risk-free rate (%)", min_value=0.1, max_value=100.0, value=5.0)
r /= 100
n = st.sidebar.number_input("Number of time steps (nodes)", min_value=1, value=10)
m = st.sidebar.number_input("Number of simulation steps (experiments)", min_value=1, value=100)
# b = st.sidebar.number_input("Distance of the barrier", min_value=0.1, value=10.0)
cp = st.sidebar.selectbox("Option", ("Call", "Put"))

st.header("Introduction")
st.markdown("Option is a right buy or sell the stock at a certain price with maturity. Since buying a beneficial "
            "option will be a difficult work without any information about the option prices, the revenue or the loss "
            "should be shown for the investors to make the profit from it. To make the buyers trade about options "
            "easier, a product that does the complicated calculations could be a useful tool. The product for this "
            "project is a dashboard that calculates the option prices including exotic options. Monte Carlo "
            "simulation is used as a formula for calculation. ")

st.header("Binomial Option Prices")

st.subheader("Introduction to the model")
st.markdown("One of the most basic methods of calculating options values is the Binomial Method. This method is "
            "reducing the possible changes in the next period’s stock price to two, an “up” move and a “down” move. "
            "Two changes in the stock price in a long period of time is unrealistic. On the other hand, we can take "
            "shorter intervals, with each interval showing two possible changes. Then, at the end of the longer "
            "period, the stock prices will be more realistic.")

# general formulas
time_step = T / n
u = np.exp(v * np.sqrt(time_step))
d = 1 / u
p = (np.exp(r * time_step) - d) / (u - d)
q = 1 - p


def binomial_lattice(S, K, r, n, call_put):
    stock_price = np.zeros((n + 1, n + 1))

    stock_price[0, 0] = S

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


binomial_price, df = binomial_lattice(S, K, r, n, cp)

st.subheader("Binomial Trees")
st.dataframe(df)
st.subheader("Binomial Price")
st.write(cp, 'price: ', round(binomial_price, 4))

st.subheader("Formulae Used")
st.latex(r"u = e^{\sigma \sqrt{t}}")
st.latex(r"d = \frac{1}{u}")
st.latex(r"S_{t+1} = S_t \cdot u")
st.latex(r"S_{t+1} = S_t \cdot d")
st.latex("C_T = max(S_T-X, 0)")
st.latex("P_T = max(X-S_T, 0)")
st.latex(r"p = \frac{e^r - d}{u - d}")
st.latex(r"C_t = e^{-r} (p \cdot Cu_{t+1} + (1 - p) \cdot Cd_{t+1})")
st.latex(r"P_t = e^{-r} (p \cdot Pu_{t+1} + (1 - p) \cdot Pd_{t+1})")
st.write("\n")

st.header("Black-Scholes Option Price")
st.subheader("Some information on the method")
st.write("")
st.subheader("Black-Scholes Price")


def black_scholes(S, K, T, r, v):
    d1 = (log(S / K) + (r + 0.5 * v ** 2) * T) / (v * sqrt(T))
    d2 = d1 - v * sqrt(T)
    if cp == "Call":
        return S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
    if cp == "Put":
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


st.write(cp, "price is: ", round(black_scholes(S, K, T, r, v), 4))

st.subheader("Formulae Used")
st.latex(r"C = N(d_1)S_t - N(d_2)Ke^{-rt}")
st.latex(r"d_1 = \frac{ln \frac{S_t}{K} + (r + \frac{\sigma^2}{2} t)}{\sigma \sqrt{t}}")
st.latex(r"d_2 = d_1 - \sigma \sqrt{t}")

st.header("Monte Carlo Simulation")

st.subheader("What is Monte Carlo Simulation for Option Pricing?")
st.markdown(
    "The calculation of option prices is an incredibly hard task, and you can never know the outcome for sure. One of "
    "the most relevant parts of the calculations of option prices is the variance. Calculating the variance as "
    "precise as possible is therefore of great importance to have a relevant outcome. This is where a Monte Carlo "
    "simulation comes in handy. A Monte Carlo simulation can execute calculations multiple times, with different "
    "initial variables. By doing this, the variance is probably more accurate than using another method, because the "
    "simulation processed many different input variables. By repeating the calculation multiple times, "
    "it will calculate multiple slightly varying option prices. In the end the tool takes the average of these "
    "calculated values. The more simulation steps the tool uses, the closer the simulation will get the option price "
    "to the actual value")


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
monte_carlo_price = round(monte_carlo()[0], 4)

st.subheader("Monte Carlo Simulation Option Price")
st.write(cp, 'price: ', monte_carlo_price)

st.subheader("Visualization of the result")

with _lock:
    fig, ax = plt.subplots()
    ax.hist(no_zeros_option_array, bins=15)
    st.pyplot(fig)

st.subheader("Formulae Used")
st.latex(r"S_{t+1} = S_t \cdot e^{d + a \cdot r}")
st.latex(r"d = (r - \frac{\sigma^2}{2}) dt")
st.latex(r"a = \sigma \cdot \sqrt{dt}")

st.header("Comparision of the methods")

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("#### Binomial Option Price")
    st.write(round(binomial_price, 4))
with col2:
    st.markdown("#### Monte Carlo Simulation Option Price")
    st.write(monte_carlo_price)
with col3:
    st.markdown("#### Black-Scholes Option Price")
    st.write(round(black_scholes(S, K, T, r, v), 4))


st.subheader("Conclusion after comparing")
st.markdown("The discrepancy in the option price from the Black-Scholes model compared to the binomial method is "
            "strictly based on the number of the time steps (nodes). Comparing the option prices achieved by the "
            "Monte Carlo simulation to the ones outputted by the Black-Scholes or the binomial model one could either "
            "notice a considerable difference between those or an agreement on the values. This is due to the "
            "quantity of the sample size chosen. Changing the number of simulations to a higher value would approach "
            "the price of the option's actual value. As the number of the run experiments increases towards infinity "
            "the price approaches the binomial and the Black-Scholes values, due to [Central Limit Theorem]("
            "https://en.wikipedia.org/wiki/Central_limit_theorem). The conclusion is that the accuracy of the "
            "binomial model depends on the number of defined nodes compared to the Monte Carlo simulation which "
            "requires a significant increment in the number of conducted simulations. ")

# rets = np.random.randn(m, T*252)*v/np.sqrt(time_step)
# st.write(rets.shape)
# traces = np.cumprod(1 + rets, 1)*S
# barrier_call = np.mean(traces[:, -1] - K * ((traces[:, -1] - K) > 0) * (np.max(traces, axis=1) < (K + b)))
# call = np.mean((traces[:, -1] - K) * ((traces[:, -1] - K) > 0))
# st.write(call)