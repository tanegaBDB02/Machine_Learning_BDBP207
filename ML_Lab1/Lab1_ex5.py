import matplotlib.pyplot as plt
import numpy as np

def derivative():
    x=np.linspace(-100,100,100)
    y=x**2
    der=2*x
    plt.plot(x,y)
    plt.plot(x,der)
    plt.show()

def main():
    derivative()

if __name__ == '__main__':
    main()

