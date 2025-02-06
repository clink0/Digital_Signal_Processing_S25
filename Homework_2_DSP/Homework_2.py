"""
Homework #2
Luke Bray
CE 332: Digital Signal Processing
Dr. Kevin Wedeward
February 2, 2025
"""

# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# %% Problem 2 Part c: Plot the stem plot of x_1[n] = cos((377*n)/f_s) for n = 0 to 10

# Define the sampling frequency
f_s = 150

# Create an array of discrete time indices from 0 to 10 (inclusive)
n = np.arange(0, 11)

# Compute the discrete-time signal x_1[n] using the cosine function.
x_1 = np.cos((377 * n) / f_s)

# Plot the discrete signal using a stem plot
plt.stem(n, x_1, basefmt="b-")
plt.xlabel('n')
plt.ylabel('x_1[n]')
plt.title('Stem plot of x_1[n] = cos(377n/150) for n = 0 to 10')
plt.grid(True)
plt.savefig('stem_plot_x_1.png', dpi=350)
plt.show()

# %% Problem 2 Part d: Compute and plot the magnitude spectrum (DTFT) of x_1[n] for 201 samples

# Define the sampling frequency
f_s = 150

# Set the number of samples to use for the DTFT computation
N = 201

# Generate a vector of frequency values from -π to π.
# Here, N-1 points are used, which provides a fine frequency resolution.
omega = np.linspace(-np.pi, np.pi, N-1)

# Create an array of discrete time indices for the 201 samples (from 0 to 200)
n = np.arange(N)

# Define x_1 as a lambda function to compute the cosine for a given n and sample frequency.
# This makes it easy to compute the signal over the vector n.
x_1 = lambda num, sample_frequency: np.cos((377 * num) / sample_frequency)

# Compute the signal x_1[n] for n = 0 to 200
x_1_n = x_1(n, f_s)

# Initialize an array to store the DTFT values for each frequency in omega.
# The array is initialized with zeros and uses a complex data type.
x_1_w = np.zeros_like(omega, dtype=complex)

# Compute the DTFT using a nested loop:
# Outer loop: iterate over each frequency value in omega.
# Inner loop: sum the contribution of each time sample n to the DTFT at that frequency.
for w in range(len(omega)):
    for i in range(N):
        # Multiply the sample x_1[n] by the complex exponential e^(-j*omega*n)
        x_1_w[w] += x_1_n[i] * np.exp(-1j * omega[w] * i)

# Compute the magnitude (absolute value) of the DTFT for each frequency
magnitude_1 = np.abs(x_1_w)

# Plot the magnitude spectrum of x_1[n]
plt.figure(figsize=(8, 4))
plt.plot(omega, magnitude_1)
plt.xlabel(r'$\omega$ (rad)')
plt.ylabel(r'$|X_1(\omega)|$')
plt.title(r'Magnitude Spectrum $|X_1(\omega)|$ for 201 samples of $x_1[n]$')
plt.grid(True)

# Define custom tick locations and labels for the x-axis
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
plt.xticks(ticks, tick_labels)

plt.savefig('magnitude_spectrum_x_1.png', dpi=350)
plt.show()

# %% Problem 4 Part c: Plot the stem plot of x_2[n] = cos((490*n)/f_s) for n = 0 to 10

# Define the sampling frequency
f_s = 150

# Create an array of discrete time indices from 0 to 10 (inclusive)
n = np.arange(0, 11)

# Compute the discrete-time signal x_2[n] using the cosine function.
x_2 = np.cos((490 * n) / f_s)

# Plot the discrete signal using a stem plot
plt.stem(n, x_2, basefmt="b-")
plt.xlabel('n')
plt.ylabel('x_2[n]')
plt.title('Stem plot of x_2[n] = cos(490n/150) for n = 0 to 10')
plt.grid(True)
plt.savefig('stem_plot_x_2.png', dpi=350)
plt.show()

# %% Problem 4 Part d: Compute and plot the magnitude spectrum (DTFT) of x_2[n] for 201 samples

# Define the sampling frequency
f_s = 150

# Set the number of samples for DTFT computation
N = 201

# Generate a vector of frequency values from -π to π using N-1 points
omega = np.linspace(-np.pi, np.pi, N-1)

# Create an array of discrete time indices for n = 0 to 200
n = np.arange(N)

# Define x_2 as a lambda function
x_2 = lambda num, sample_frequency: np.cos((490 * num) / sample_frequency)

# Compute the signal x_2[n] for n = 0 to 200
x_2_n = x_2(n, f_s)

# Initialize an array to store the DTFT values of x_2[n] with complex data type
x_2_w = np.zeros_like(omega, dtype=complex)

# Compute the DTFT using nested loops over frequency and time indices
for w in range(len(omega)):
    for i in range(N):
        # Multiply x_2[n] by the complex exponential e^(-j*omega*n) and sum over n
        x_2_w[w] += x_2_n[i] * np.exp(-1j * omega[w] * i)

# Compute the magnitude (absolute value) of the DTFT for each frequency
magnitude_2 = np.abs(x_2_w)

# Plot the magnitude spectrum of x_2[n]
plt.figure(figsize=(8, 4))
plt.plot(omega, magnitude_2)
plt.xlabel(r'$\omega$ (rad)')
plt.ylabel(r'$|X_2(\omega)|$')
plt.title(r'Magnitude Spectrum $|X_2(\omega)|$ for 201 samples of $x_2[n]$')
plt.grid(True)

# Set custom tick locations and labels for the x-axis
ticks = [-np.pi, -np.pi/2, 0, np.pi/2, np.pi]
tick_labels = [r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$']
plt.xticks(ticks, tick_labels)

plt.savefig('magnitude_spectrum_x_2.png', dpi=350)
plt.show()