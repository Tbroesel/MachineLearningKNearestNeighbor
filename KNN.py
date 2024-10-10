import numpy as np

def euclidean_distance_optimized(X_train, X_test):
    """
    Vectorized computation of Euclidean distances between training and test points.
    """
    distances = np.sqrt(np.sum((X_train[:, np.newaxis] - X_test) ** 2, axis=2))
    return distances


def k_nearest_neighbors(X_train, y_train, X_test, k, regression=False, sigma=None):
    """
    Optimized k-NN classifier/regressor for a given test set, using vectorized distance calculation.
    """
    # Ensure sigma is not None for regression
    if regression and sigma is None:
        sigma = 1.0  # Set a default value for sigma

    predictions = []

    # Vectorized distance calculation
    distances = euclidean_distance_optimized(X_train, X_test)

    for i in range(len(X_test)):
        # Get distances for the current test point
        current_distances = distances[:, i]

        # Sort by distance and get the k nearest neighbors
        nearest_indices = np.argsort(current_distances)[:k]
        neighbors = y_train[nearest_indices]

        if regression:
            weights = np.exp(-current_distances[nearest_indices]**2 / (2 * sigma**2))
            weighted_avg = np.dot(weights, neighbors) / np.sum(weights)
            predictions.append(weighted_avg)
        else:
            # Majority voting for classification
            prediction = max(set(neighbors), key=list(neighbors).count)
            predictions.append(prediction)

    return np.array(predictions)


def edited_knn(X_train, y_train, k, regression=False, error_threshold=None):
    """
    Edited k-NN, which removes misclassified points from the training set.
    For regression, it uses an error threshold to define if a prediction is incorrect.
    """
    edited_X, edited_y = [], []

    # Ensure a default error_threshold is provided if missing during regression
    if regression and error_threshold is None:
        error_threshold = 0.1  # Default error threshold for regression

    for i in range(len(X_train)):
        # Leave one out and test on the ith point
        X_edited = np.concatenate((X_train[:i], X_train[i+1:]), axis=0)
        y_edited = np.concatenate((y_train[:i], y_train[i+1:]), axis=0)

        prediction = k_nearest_neighbors(X_edited, y_edited, [X_train[i]], k, regression=regression)[0]

        if regression:
            # Check if the error is within the threshold for regression
            if abs(prediction - y_train[i]) <= error_threshold:
                edited_X.append(X_train[i])
                edited_y.append(y_train[i])
        else:
            # Keep the point if correctly classified for classification tasks
            if prediction == y_train[i]:
                edited_X.append(X_train[i])
                edited_y.append(y_train[i])

    return np.array(edited_X), np.array(edited_y)


def kmeans_clustering(X, num_clusters, max_iterations=100):
    """
    Apply k-means clustering to the data and return cluster centroids as a reduced dataset.
    """
    np.random.seed(42)
    centroids = X[np.random.choice(len(X), num_clusters, replace=False)]

    for _ in range(max_iterations):
        clusters = [[] for _ in range(num_clusters)]

        # Assign each data point to the closest centroid
        for point in X:
            distances = [euclidean_distance_optimized(point, centroid) for centroid in centroids]
            closest_centroid_idx = np.argmin(distances)
            clusters[closest_centroid_idx].append(point)

        new_centroids = [np.mean(cluster, axis=0) if len(cluster) > 0 else centroids[i]
                         for i, cluster in enumerate(clusters)]

        if np.allclose(centroids, new_centroids, atol=1e-6):
            break

        centroids = new_centroids

    return np.array(centroids)

