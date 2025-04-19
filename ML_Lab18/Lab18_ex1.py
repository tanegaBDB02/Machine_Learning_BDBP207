import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from mlxtend.plotting import plot_decision_regions


data = {
    'x1': [6, 6, 8, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 14],
    'x2': [5, 9, 6, 8, 10, 2, 5, 10, 13, 5, 8, 6, 11, 4, 8],
    'Label': ['Blue', 'Blue', 'Red', 'Red', 'Red', 'Blue', 'Red', 'Red', 'Blue',
              'Red', 'Red', 'Red', 'Blue', 'Blue', 'Blue']
}
df = pd.DataFrame(data)

X1 = df['x1']
X2 = df['x2']
X = df[['x1', 'x2']].values

le = LabelEncoder()
y = le.fit_transform(df['Label'])

print("X1:\n", X1)
print("\nX2:\n", X2)
print("\nX matrix:\n", X)
print("\nEncoded Labels (y):\n", y)

clf_rbf = SVC(kernel='rbf', gamma='scale')
clf_rbf.fit(X, y)


clf_poly = SVC(kernel='poly', degree=3)
clf_poly.fit(X, y)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plot_decision_regions(X, y, clf=clf_rbf, legend=2)
plt.title("RBF Kernel")

plt.subplot(1,2,2)
plot_decision_regions(X, y, clf=clf_poly, legend=2)
plt.title("Polynomial Kernel")
plt.tight_layout()

plt.show()

print("RBF Accuracy:", clf_rbf.score(X, y))
print("Poly Accuracy:", clf_poly.score(X, y))