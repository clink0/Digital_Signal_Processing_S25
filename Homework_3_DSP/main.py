"""
Luke Bray
CE 332
February 10, 2025
"""

# Imports
import numpy as np
import matplotlib.pyplot as plt

# %% Problem 3 part a

N_impulse = 10
h = np.zeros(N_impulse + 1)
x_impulse = np.zeros(N_impulse + 1)
x_impulse[0] = 1

for n in range(N_impulse + 1):
    if n == 0:
        h[n] = x_impulse[n]
    else:
        h[n] = 0.6 * h[n - 1] + x_impulse[n]


# %% Problem 3 part b

N_cosine = 35
x_cosine = np.cos(2 * np.pi * np.arange(N_cosine + 1) / 10)
y_recursive = np.zeros(N_cosine + 1)

for n in range(N_cosine + 1):
    if n == 0:
        y_recursive[n] = x_cosine[n]
    else:
        y_recursive[n] = 0.6 * y_recursive[n - 1] + x_cosine[n]

# %% Problem 3 part c

# y_convolution = np.convolve(x_cosine, h)[:N_cosine + 1]

y_manual_convolution = np.zeros(N_cosine + 1)

for n in range(N_cosine + 1):
    for k in range(len(h)):
        if n - k >= 0:
            y_manual_convolution[n] += h[k] * x_cosine[n - k]


# %% Problem 3 part d
plt.figure(figsize=(10, 6))
plt.stem(y_recursive, label="Recursive", markerfmt="ro")
plt.title("Comparison of Recursive and Convolution Outputs")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 6))
plt.stem(y_manual_convolution, label="Convolution", markerfmt="rd")
plt.title("Comparison of Recursive and Convolution Outputs")
plt.xlabel("n")
plt.ylabel("y[n]")
plt.legend()
plt.grid()
plt.show()


print('The recursive and convolution methods produced nearly identical graphs.\n'
      'However, there are some very slight differences.  Given that the impulse response\n'
      'was not computed infinitely, a small error was introduced.  The error is not great \n'
      'enough that we can pick up on it visually on the graph but it is worth noting.\n'
      'I played around a bit with the N_impulse parameter and noticed that when it \n'
      'is set at a low number (ie 3) the graphs begin to look more different.  This is because when \n '
      'the impulse response is computed for only a few numbers the error is much greater. \n'
      'Otherwise these are just two methods of computing the same thing.')