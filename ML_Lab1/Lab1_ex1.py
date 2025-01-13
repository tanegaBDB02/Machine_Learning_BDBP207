import numpy as np

def transpose(A):
    trans = np.transpose(A)
    mult = np.dot(trans, A)
    return mult

def main():
    A = np.array([[1, 2, 3], [4, 5, 6]])
    Trans = transpose(A)
    print(Trans)

if __name__ == '__main__':
    main()
