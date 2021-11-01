import numpy as np

S = 100
K = 105
T = 1
r = 0.05
v = 0.1
m = 100
n = 10

time_step = T / n
drift = (r - (v ** 2) / 2) * time_step
a = v * np.sqrt(time_step)
x = np.random.normal(0, 1, (m, n))
# print(x)

Smat = np.zeros((m, n))
Smat[:, 0] += S

for i in range(1, n):
    Smat[:, i] += Smat[:, i - 1] * np.exp(drift + a * x[:, i])

q = Smat[:, -1] - K
for i in range(len(q)):
    if q[i] < 0:
        q[i] = 0
    else:
        q[i] = q[i]

p = K - Smat[:, -1]
for i in range(len(p)):
    if p[i] < 0:
        p[i] = 0
    else:
        p[i] = p[i]

payoff_call = np.mean(q)
payoff_put = np.mean(p)

# print(Smat)
# print(payoff_call)
# print(payoff_put)

call = payoff_call*np.exp(-r*T)
put = payoff_put*np.exp(-r*T)

# print(call)
# print(put)

new_p = np.delete(p, np.where(p==0.))
print(new_p)