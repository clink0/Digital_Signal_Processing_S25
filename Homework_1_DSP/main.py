"""
Homework #1 Complex Numbers Review
Luke Bray
CE 332: Digital Signal Processing
Dr. Kevin Wedeward
January 17, 2025
"""

#imports
import numpy as np
import matplotlib.pyplot as plt

#function to print complex numbers in all three forms
def print_complex(lbl, z):
    print(f"{lbl} = {np.real(z):.2f} + j({np.imag(z):.2f})")              #rectangular
    print(f"   = {np.abs(z):.2f} /_({np.angle(z) * 180 / np.pi:.2f}deg)") #phasor
    print(f"   = {np.abs(z):.2f} e^(j({np.angle(z):.2f}))")               #complex exponential

#set z1 and z2
z1 = -2 + 1j
z2 = np.sqrt(5) * np.exp(1j * (3 * np.pi/4))

print("\n*****************************")
print("         Problem 3")
print("*****************************")

print_complex("z1", z1)
print("\n")
print_complex("z2", z2)
print("\n")

print("*****************************")
print("          Part a")
print("*****************************")

print("z1 + z2 = z3")
z3 = z1 + z2
print_complex("z3", z3)
print("\n")

print("*****************************")
print("          Part b")
print("*****************************")
print("\nz1 * z2 = z3")
z3 = z1 * z2
print_complex("z3", z3)
print("\n")

print("*****************************")
print("          Part c")
print("*****************************")
print("\nz1 / z2 = z3")
z3 = z1 / z2
print_complex("z3", z3)