from dataPreProcess import dataPreProcess, cross_validate_manual, stratified_k_fold_cross_validation
from KNN import k_nearest_neighbors, edited_knn
import numpy as np
from pathlib import Path

def main(dataset_path, regression=True):
    # Load and preprocess data
    X, y = dataPreProcess(dataset_path)

    # List of k values to try
    k_values = [1, 3, 5, 7, 9]

    # Cross-validate and find the best k value using Edited k-NN
    if regression:
        best_k = cross_validate_manual(X, y, k_values, regression=True, sigma=1.0, error_threshold=0.1)
    else:
        best_k = cross_validate_manual(X, y, k_values)

    print(f"Best k value: {best_k}")

    # 10-fold stratified cross-validation to report accuracy or MSE for each fold
    folds = stratified_k_fold_cross_validation(X, y, num_folds=10, regression=regression)

    print("\nFinal evaluation on each fold:")

    for i in range(10):
        test_set = folds[i]
        train_set = np.concatenate(folds[:i] + folds[i+1:])

        X_train, y_train = train_set[:, :-1], train_set[:, -1]
        X_test, y_test = test_set[:, :-1], test_set[:, -1]

        # Apply Edited k-NN on training set
        X_train_edited, y_train_edited = edited_knn(X_train, y_train, best_k, regression=regression)

        # Run k-NN on the edited training set
        predictions = k_nearest_neighbors(X_train_edited, y_train_edited, X_test, best_k, regression=regression, sigma=1.0)

        if regression:
            mse = np.mean((predictions - y_test) ** 2)
            print(f"Fold {i+1}: Mean Squared Error (MSE) = {mse}")
        else:
            accuracy = np.mean(predictions == y_test)
            print(f"Fold {i+1}: Accuracy = {accuracy}")

if __name__ == "__main__":
    dataset_path = r"C:\Users\camde\IdeaProjects\MachineLearningKNearestNeighbor\data\breast-cancer-wisconsin.data"  # Specify the dataset location here
    main(dataset_path)

