from dataPreProcess import dataPreProcess, cross_validate_manual, stratified_k_fold_cross_validation
from KNN import k_nearest_neighbors, edited_knn
import numpy as np

def main(dataset_path, regression=False):
    # Load and preprocess data
    X, y = dataPreProcess(dataset_path)

    # List of k values to try
    k_values = [1, 3, 5, 7, 9]
    sigma_values = [.6, .8, 1., 1.2, 1.4]

    # Cross-validate and find the best k value
    if regression:
        # Regression: Expect both best_k and best_sigma
        best_k, best_sigma = cross_validate_manual(X, y, k_values, regression=True, sigma=sigma_values, error_threshold=0.1)
        print(f"Best k and sigma value: {best_k} with {best_sigma}")
    else:
        # Classification: Only best_k is returned
        best_k = cross_validate_manual(X, y, k_values)
        print(f"Best k: {best_k}")
        best_sigma = None  # No sigma for classification

    # Cast best_k to int, ensuring it's not a tuple
    if isinstance(best_k, tuple):
        best_k = best_k[0]  # Unpack if it's mistakenly a tuple

    # Perform stratified 10-fold cross-validation and evaluation
    folds = stratified_k_fold_cross_validation(X, y, num_folds=10, regression=regression)

    print("\nFinal evaluation on each fold:")
    avg_performance = 0
    for i in range(10):
        test_set = folds[i]
        train_set = np.concatenate(folds[:i] + folds[i+1:])

        X_train, y_train = train_set[:, :-1], train_set[:, -1]
        X_test, y_test = test_set[:, :-1], test_set[:, -1]

        # Apply Edited k-NN on training set
        X_train_edited, y_train_edited = edited_knn(X_train, y_train, int(best_k), regression=regression)

        # Run k-NN on the edited training set
        predictions = k_nearest_neighbors(X_train_edited, y_train_edited, X_test, int(best_k), regression=regression, sigma=best_sigma)

        if regression:
            mse = np.mean((predictions - y_test) ** 2)
            avg_performance += mse
            print(f"Fold {i+1}: Mean Squared Error (MSE) = {mse}")
        else:
            accuracy = np.mean(predictions == y_test)
            avg_performance += accuracy
            print(f"Fold {i+1}: Accuracy = {accuracy}")

    # Average the performance over all folds
    avg_performance /= 10
    print(f"Final Average Performance = {avg_performance}")


if __name__ == "__main__":
    dataset_path = r"C:\Users\camde\IdeaProjects\MachineLearningKNearestNeighbor\data\glass.data"  # Specify the dataset location here
    #dataset_path = r"C:\Users\tyler\OneDrive\Documents\GitHub\MachineLearningKNearestNeighbor\data\breast-cancer-wisconsin.data"
    main(dataset_path)

