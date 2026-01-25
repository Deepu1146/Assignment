# LAB 3 : A1 to A14 (CMU‑MOSI numeric data)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import minkowski

data = pd.read_csv("cmu_mosi_numeric.csv")
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# A1
def dot_product(A, B): return sum(a*b for a, b in zip(A, B))
def euclidean_norm(A): return np.sqrt(sum(a*a for a in A))

# A2
def mean_vector(X): return np.mean(X, axis=0)
def variance_vector(X): return np.var(X, axis=0)
def std_vector(X): return np.std(X, axis=0)
def class_centroid(X, y, label): return X[y == label].mean(axis=0)

# A3
plt.hist(X[:, 0], bins=10)
plt.show()

# A4, A5
def minkowski_own(A, B, p): return sum(abs(a-b)**p for a, b in zip(A, B))**(1/p)
p_vals = range(1, 11)
plt.plot(p_vals, [minkowski_own(X[0], X[1], p) for p in p_vals])
plt.show()

# A6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# A7, A8, A9
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
print("Accuracy:", knn.score(X_test, y_test))
y_pred = knn.predict(X_test)

# A10
def knn_predict(Xtr, ytr, x, k):
    d = [(euclidean_norm(x-xi), yi) for xi, yi in zip(Xtr, ytr)]
    d.sort(key=lambda z: z[0])
    return max([l for _, l in d[:k]], key=[l for _, l in d[:k]].count)

y_pred_own = np.array([knn_predict(X_train, y_train, x, 3) for x in X_test])

# A11
acc = []
for k in range(1, 12):
    preds = np.array([knn_predict(X_train, y_train, x, k) for x in X_test])
    acc.append(np.mean(preds == y_test))
plt.plot(range(1, 12), acc)
plt.show()

# A12, A13, A14
cm = confusion_matrix(y_test, y_pred_own)
tp = cm[1,1]; fp = cm[0,1]; fn = cm[1,0]
precision = tp / (tp + fp + 1e-9)
recall = tp / (tp + fn + 1e-9)
f1 = 2 * precision * recall / (precision + recall + 1e-9)

print("Confusion Matrix:\n", cm)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)
