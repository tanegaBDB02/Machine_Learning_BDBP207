import matplotlib.pyplot as plt
import numpy as np

def lin_eq():
    x= np.linspace(-100,100,100)
    y= (2*x)+3
    plt.plot(x,y)
    plt.show()

def main():
    lin_eq()

if __name__ == "__main__":
    main()