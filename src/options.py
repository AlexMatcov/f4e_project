import numpy as np
import pandas as pd

def binomial_lattice(S0, K, r, v, T, n, call_put, exercise_policy):
    time_step = T / n

    # Compute u and d
    """ Fill in appropriate formulas"""
    u = np.exp(v * np.sqrt(time_step))
    d = 1 / u

    # Compute p and q
    """ Fill in appropriate formulas"""
    p = (np.exp(r * time_step) - d) / (u - d)
    q = 1 - p

    # Create empty matrix for stock prices
    stock_price = np.zeros((n + 1, n + 1))

    # Set initial stock price
    stock_price[0, 0] = S0

    # Fill matrix with stock prices per time step
    if exercise_policy == "European":
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
            """Fill in appropriate formula"""
            option_value[n, i] = max(0, stock_price[n, i] - K)
        elif call_put == 'Put':
            """Fill in appropriate formula"""
            option_value[n, i] = max(0, K - stock_price[n, i])

    # Compute discount factor per time step
    """Fill in appropriate formula"""
    discount = np.exp(-r * time_step)

    # Recursively compute option value at time 0
    for i in range(n - 1, -1, -1):
        for j in range(i + 1):
            """ Fill in appropriate formulas for the different option types"""
            option_value[i, j] = discount * (p * option_value[i + 1, j] + q * option_value[i + 1, j + 1])

    return option_value[0, 0], df_stock_price


# Test case: the following settings should yield an option price of 4.04
S = 100
K = 105
v = 0.1
T = 1
r = 0.05
n = 10
call_put = 'Call'
exercise_policy = 'European'

binomial_price, df = binomial_lattice(S, K, r, v, T, n, call_put, exercise_policy)

print('Binomial lattice price: %.2f' % binomial_price)
df