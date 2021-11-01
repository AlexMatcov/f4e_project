import numpy as np
import pandas as pd

class Calculate:

    def __init__(self, S, K, v, T, r, n, cp, ep):
        self.S = S
        self.K = K
        self.v = v
        self.T = T
        self.r = r
        self.n = n
        self.cp = cp
        self.ep = ep

    def binomial_lattice(self):

        time_step = self.T / self.n

        u = np.exp(self.v * np.sqrt(time_step))
        d = 1 / u
        p = (np.exp(self.r * time_step) - d) / (u - d)
        q = 1 - p

        stock_price = np.zeros((self.n + 1, self.n + 1))

        stock_price[0, 0] = self.S

        if self.ep == "European":
            for i in range(1, self.n + 1):
                stock_price[i, 0] = stock_price[i - 1, 0] * u
                for j in range(1, i + 1):
                    stock_price[i, j] = stock_price[i - 1, j - 1] * d

        df_stock_price = pd.DataFrame(data=stock_price)
        df_stock_price = df_stock_price.T

        option_value = np.zeros((self.n + 1, self.n + 1))

        for i in range(self.n + 1):
            if self.cp == 'Call':
                option_value[self.n, i] = max(0, stock_price[self.n, i] - self.K)
            elif self.cp == 'Put':
                option_value[self.n, i] = max(0, self.K - stock_price[self.n, i])

        discount = np.exp(-self.r * time_step)

        for i in range(self.n - 1, -1, -1):
            for j in range(i + 1):
                option_value[i, j] = discount * (p * option_value[i + 1, j] + q * option_value[i + 1, j + 1])

        return option_value[0, 0], df_stock_price


