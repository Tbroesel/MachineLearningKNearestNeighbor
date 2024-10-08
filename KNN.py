import numpy as np
import math
from collections import Counter
from math import sqrt

# Distance calculation (Euclidean)
def euclidean_distance(x1, x2):
    return sqrt(np.sum((np.array(x1) - np.array(x2))**2))

# k-NN algorithm implementation
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test, task="classification"):
        predictions = [self._predict(x, task) for x in X_test]
        return np.array(predictions)

    def _predict(self, x, task):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Get the nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if task == "classification":
            # Return the most common class label (classification)
            return Counter(k_nearest_labels).most_common(1)[0][0]
        else:
            # Return the average of k neighbors (regression)
            return np.mean(k_nearest_labels)


class EditedKNN(KNN):
    def __init__(self, k=3, threshold=0.1):
        super().__init__(k)
        self.threshold = threshold

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        # Remove noisy examples based on threshold
        self.X_train, self.y_train = self._edit_examples(self.X_train, self.y_train)

    def _edit_examples(self, X_train, y_train):
        X_cleaned = []
        y_cleaned = []

        for i, x in enumerate(X_train):
            # Use k-NN to predict the label of each example
            neighbors = np.argsort([euclidean_distance(x, x_train) for x_train in X_train if x_train != x])[:self.k]
            k_nearest_labels = [y_train[j] for j in neighbors]
            predicted_label = Counter(k_nearest_labels).most_common(1)[0][0]

            # If prediction is close enough to the true label, keep the example
            if abs(predicted_label - y_train[i]) < self.threshold:
                X_cleaned.append(x)
                y_cleaned.append(y_train[i])

        return X_cleaned, y_cleaned


class KMeans:
    def __init__(self, k_clusters=3, max_iters=100):
        self.k_clusters = k_clusters
        self.max_iters = max_iters

    def fit(self, X):
        # Convert X to a NumPy array if it isn't already
        X = np.array(X)

        # Randomly initialize centroids from the dataset
        self.centroids = X[np.random.choice(X.shape[0], self.k_clusters, replace=False)]

        for _ in range(self.max_iters):
            # Assign clusters
            clusters = self._assign_clusters(X)
            # Recompute centroids
            new_centroids = self._compute_centroids(X, clusters)
            # Compare centroids using np.array_equal
            if np.array_equal(new_centroids, self.centroids):
                break
            self.centroids = new_centroids
        return self.centroids

    def _assign_clusters(self, X):
        clusters = []
        for x in X:
            distances = [euclidean_distance(x, centroid) for centroid in self.centroids]
            clusters.append(np.argmin(distances))
        return clusters

    def _compute_centroids(self, X, clusters):
        return [np.mean([X[i] for i in range(len(X)) if clusters[i] == cluster], axis=0) for cluster in range(self.k_clusters)]


def gaussian_kernel(distance, sigma=1.0):
    return math.exp(-distance**2 / (2 * sigma**2))


class KNNWithRBF(KNN):
    def __init__(self, k=3, sigma=1.0):
        super().__init__(k)
        self.sigma = sigma

    def _predict(self, x, task):
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        if task == "classification":
            return Counter(k_nearest_labels).most_common(1)[0][0]
        else:
            weights = [gaussian_kernel(distances[i], self.sigma) for i in k_indices]
            return np.dot(k_nearest_labels, weights) / np.sum(weights)


# Example usage of k-NN
if __name__ == "__main__":
    # Dummy data (simple 2D points)
    X_train = [[2, 3], [3, 5], [5, 8], [7, 7], [8, 10]]
    y_train_class = [0, 1, 1, 0, 1]  # Classification labels
    y_train_reg = [2.5, 3.6, 4.2, 5.8, 6.1]  # Regression values

    knn = KNN(k=3)

    # Classification example
    knn.fit(X_train, y_train_class)
    X_test = [[6, 7], [5, 6]]
    print("Classification Predictions:", knn.predict(X_test, task="classification"))

    # Regression example
    knn.fit(X_train, y_train_reg)
    print("Regression Predictions:", knn.predict(X_test, task="regression"))


# Example usage of Edited k-NN
if __name__ == "__main__":
    enn = EditedKNN(k=3, threshold=0.1)

    # Classification example
    enn.fit(X_train, y_train_class)
    print("Edited k-NN Classification Predictions:", enn.predict(X_test, task="classification"))

    # Regression example
    enn.fit(X_train, y_train_reg)
    print("Edited k-NN Regression Predictions:", enn.predict(X_test, task="regression"))


# Example usage of k-Means
if __name__ == "__main__":
    kmeans = KMeans(k_clusters=3)
    centroids = kmeans.fit(X_train)
    print("Centroids from k-Means:", centroids)


# Example usage for regression with RBF
if __name__ == "__main__":
    knn_rbf = KNNWithRBF(k=3, sigma=1.0)
    knn_rbf.fit(X_train, y_train_reg)
    print("RBF Regression Predictions:", knn_rbf.predict(X_test, task="regression"))