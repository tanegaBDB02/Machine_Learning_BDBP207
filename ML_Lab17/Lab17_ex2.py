from ML_Lab17.Lab17_ex1 import Transform
import numpy as np

def high_dim(x1,x2):
    phi_x1 = Transform(x1[0], x1[1]).flatten()
    phi_x2 = Transform(x2[0], x2[1]).flatten()

    dot_prod=np.dot(phi_x1,phi_x2)
    print(f"Dot product in transformed space: {dot_prod:.4f}")

    return phi_x1, phi_x2

def polynomial_kernel(a,b):
    return (a[0] ** 2 * b[0] ** 2 + 2 * a[0] * b[0] * a[1] * b[1] + a[1] ** 2 * b[1] ** 2)

def main():
    x1 = [3, 6]
    x2 = [10, 10]
    high_dim(x1,x2)
    for i in range(len(x1)):
        a = [x1[i], x2[i]]
        b = [x1[i], x2[i]]
        print(f"Point: ({x1[i]}, {x2[i]})")
        print(f"  Kernel: {polynomial_kernel(a, b)}")


if __name__ == "__main__":
    main()


