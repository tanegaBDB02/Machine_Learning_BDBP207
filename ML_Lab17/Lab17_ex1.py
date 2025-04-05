import numpy as np
import matplotlib.pyplot as plt

def Transform(x1,x2):
    return np.vstack([x1**2, np.sqrt(2)*x1*x2, x2**2]).T

def implement():
    x1=np.array([1,1,2,3,6,9,13,18,3,6,6,9,10,11,12,16])
    x2=np.array([13,18,9,6,3,2,1,1,15,6,11,5,10,5,6,3])
    labels = np.array(["Blue"] * 8 + ["Red"] * 8,dtype=object)

    transformed_data=Transform(x1,x2)

    for i in range(len(x1)):
        print(f"Original: ({x1[i]}, {x2[i]}) → Transformed: {transformed_data[i]} → Label: {labels[i]}")

    colors = ['b' if label == "Blue" else 'r' for label in labels]

    return transformed_data, colors

def plot(transformed_data, colors):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(transformed_data[:, 0], transformed_data[:, 1], transformed_data[:, 2], c=colors, s=50)

    ax.set_xlabel("phi(x1) = x1^2")
    ax.set_ylabel("phi(x1, x2) = sqrt(2) * x1 * x2")
    ax.set_zlabel("phi(x2) = x2^2")
    ax.set_title("Transformed Data in 3D")

    plt.show()

if __name__ == "__main__":
    transformed_data, colors = implement()
    plot(transformed_data, colors)



