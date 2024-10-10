
from dataPreProcess import dataPreProcess, cross_validate_manual, stratified_k_fold_cross_validation
from KNN import k_nearest_neighbors, edited_knn
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

def main(dataset_path, regression=False):
    # Load and preprocess data
    X, y = dataPreProcess(dataset_path)

    # List of k values to try
    k_values = [1, 3, 5, 7, 9]
    sigma_values = [.6, .8, 1., 1.2, 1.4]
    # Cross-validate and find the best k value using Edited k-NN
    if regression:
        best_k, best_s = cross_validate_manual(X, y, k_values, regression=True, sigma=sigma_values, error_threshold=0.1)
        print(f"Best k and sigma value: {best_k} with {best_s}")

    else:
        best_k = cross_validate_manual(X, y, k_values)
        print(f"Best k: {best_k}")
    

    # 10-fold stratified cross-validation to report accuracy or MSE for each fold
    folds = stratified_k_fold_cross_validation(X, y, num_folds=10, regression=regression)

    print("\nFinal evaluation on each fold:")
    Avg = 0
    for i in range(10):
        test_set = folds[i]
        train_set = np.concatenate(folds[:i] + folds[i+1:])

        X_train, y_train = train_set[:, :-1], train_set[:, -1]
        X_test, y_test = test_set[:, :-1], test_set[:, -1]

        # Apply Edited k-NN on training set
        X_train_edited, y_train_edited = edited_knn(X_train, y_train, best_k, regression=regression)

        # Run k-NN on the edited training set
        predictions = k_nearest_neighbors(X_train_edited, y_train_edited, X_test, best_k, regression=regression, sigma=best_s)

        if regression:
            mse = np.mean((predictions - y_test) ** 2)
            Avg += mse
            print(f"Fold {i+1}: Mean Squared Error (MSE) = {mse}")
        else:
            accuracy = np.mean(predictions == y_test)
            Avg += accuracy
            print(f"Fold {i+1}: Accuracy = {accuracy}")
    print(f"Final Average = {Avg}")


    



if __name__ == "__main__":
    #dataset_path = r"C:\Users\camde\IdeaProjects\MachineLearningKNearestNeighbor\data\breast-cancer-wisconsin.data"  # Specify the dataset location here
    dataset_path = r"C:\Users\tyler\OneDrive\Documents\GitHub\MachineLearningKNearestNeighbor\data\glass.data"
    main(dataset_path)

