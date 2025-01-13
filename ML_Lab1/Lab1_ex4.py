import matplotlib.pyplot as plt
import numpy as np

def gaussian_pdf(x, mean, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / sigma) ** 2)


mean = 0
sigma = 15
x = np.linspace(-100, 100,100)

y = gaussian_pdf(x, mean, sigma)

plt.plot(x, y)
plt.show()
