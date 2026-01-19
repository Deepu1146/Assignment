import numpy as np
import matplotlib.pyplot as plt

# =========================================================
# A1: VECTOR OPERATIONS
# =========================================================

def dot_product(vec_a, vec_b):
    """
    Calculate dot product of two vectors
    """
    return sum(a * b for a, b in zip(vec_a, vec_b))


def euclidean_norm(vec):
    """
    Calculate Euclidean norm (length) of a vector
    """
    return (sum(x ** 2 for x in vec)) ** 0.5


# =========================================================
# A2: INTRACLASS SPREAD & INTERCLASS DISTANCE
# =========================================================

def calculate_mean(data):
    """
    Calculate mean of a list or 1D array
    """
    return sum(data) / len(data)


def calculate_variance(data):
    """
    Calculate variance of a list or 1D array
    """
    mean_val = calculate_mean(data)
    return sum((x - mean_val) ** 2 for x in data) / len(data)


def calculate_std(data):
    """
    Calculate standard deviation
    """
    return calculate_variance(data) ** 0.5


def dataset_statistics(matrix):
    """
    Calculate mean and standard deviation for each feature column
    """
    means = []
    stds = []

    for col in range(matrix.shape[1]):
        feature = matrix[:, col]
        means.append(calculate_mean(feature))
        stds.append(calculate_std(feature))

    return np.array(means), np.array(stds)


def interclass_distance(centroid1, centroid2):
    """
    Calculate Euclidean distance between two centroids
    """
    return np.linalg.norm(centroid1 - centroid2)


# =========================================================
# A3: HISTOGRAM, MEAN & VARIANCE
# =========================================================

def compute_mean_variance(data):
    """
    Calculate mean and variance of a feature
    """
    mean_val = sum(data) / len(data)
    variance_val = sum((x - mean_val) ** 2 for x in data) / len(data)
    return mean_val, variance_val


def plot_histogram(feature, bins=5):
    """
    Plot histogram for a feature
    """
    plt.hist(feature, bins=bins)
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.title("Histogram of Selected Feature")
    plt.show()


# =========================================================
# MAIN PROGRAM
# =========================================================

if __name__ == "__main__":

    # ---------------- A1 ----------------
    A = np.array([1, 2, 3])
    B = np.array([4, 5, 6])

    manual_dot = dot_product(A, B)
    numpy_dot = np.dot(A, B)

    manual_norm = euclidean_norm(A)
    numpy_norm = np.linalg.norm(A)

    print("A1 RESULTS")
    print("Manual Dot Product:", manual_dot)
    print("NumPy Dot Product:", numpy_dot)
    print("Manual Euclidean Norm:", manual_norm)
    print("NumPy Euclidean Norm:", numpy_norm)

    # ---------------- A2 ----------------
    class1 = np.array([[1, 2], [2, 3], [3, 4]])
    class2 = np.array([[6, 7], [7, 8], [8, 9]])

    centroid1 = class1.mean(axis=0)
    centroid2 = class2.mean(axis=0)

    spread1 = class1.std(axis=0)
    spread2 = class2.std(axis=0)

    distance = interclass_distance(centroid1, centroid2)

    print("\nA2 RESULTS")
    print("Class 1 Centroid:", centroid1)
    print("Class 2 Centroid:", centroid2)
    print("Class 1 Spread:", spread1)
    print("Class 2 Spread:", spread2)
    print("Interclass Distance:", distance)

    # ---------------- A3 ----------------
    feature = np.array([10, 12, 15, 18, 20, 22, 25, 30, 35, 40])

    mean_val, variance_val = compute_mean_variance(feature)

    print("\nA3 RESULTS")
    print("Feature Mean:", mean_val)
    print("Feature Variance:", variance_val)

    plot_histogram(feature, bins=5)
