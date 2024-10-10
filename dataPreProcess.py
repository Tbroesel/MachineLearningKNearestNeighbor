import numpy as np
import pandas as pd
import multiprocessing as mp
from KNN import k_nearest_neighbors, edited_knn
import matplotlib.pyplot as plt


def dataPreProcess(file_path, normalize_y=True):
    """
    Load and preprocess data from a CSV file, handling missing values and
    converting categorical features to numeric ones using factorization.
    Optionally normalize target values y.
    """
    # Load data into a pandas DataFrame
    data = pd.read_csv(file_path, header=None)
    
    # Process each column (Handle categorical features)
    for row, element in enumerate(data.iloc[0]):
        if isinstance(element, str):  # If the element is a string (categorical data)
            data[row] = pd.factorize(data[row])[0]  # Convert the entire column to numeric (factorize)

    # Fill missing values (for numeric columns)
    data.fillna(data.mean(numeric_only=True).round(0), inplace=True)  # Replace missing numeric values with the mean

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)  # Shuffle the data

    # Split into features (X) and labels (y)
    X = data.iloc[:, :-1].values  # All columns except the last one (features)
    y = data.iloc[:, -1].values   # The last column contains labels/targets

    # Normalize the target values if requested
    if normalize_y:
        y = (y - np.min(y)) / (np.max(y) - np.min(y))  # Normalize to [0, 1]

    # Standardize the features: Only divide by std if std != 0
    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)
    stds[stds == 0] = 1  # Replace 0 std with 1 to avoid division by zero
    X = (X - means) / stds

    rowCount = 0
    for row, element in enumerate(data.iloc[0]):
        rowCount+=1
    #for count in range(rowCount):
        #plt.scatter(data.iloc[:, count], data.iloc[:, -1], alpha=0.5)
        #plt.show()

    return X, y


def stratified_k_fold_cross_validation(X, y, num_folds=10, regression=False):
    """
    Perform stratified 10-fold cross-validation.
    For regression, the data is stratified by sorting based on the target values.
    """
    data = np.column_stack((X, y))
    if regression:
        data = data[data[:, -1].argsort()]  # Sort by the target values

    fold_size = len(X) // num_folds
    folds = []

    for i in range(num_folds):
        fold = data[i::num_folds]
        folds.append(fold)

    return folds


def process_fold(fold_data):
    """Function to process a single fold of cross-validation."""
    X_train, y_train, X_test, y_test, k, regression, sigma, error_threshold = fold_data

    # Ensure sigma is not None for regression
    if regression and sigma is None:
        sigma = 1.0  # Set a default sigma if not provided

    # Apply Edited k-NN on training set
    X_train_edited, y_train_edited = edited_knn(X_train, y_train, k, regression=regression, error_threshold=error_threshold)

    # Run k-NN on the edited training set
    predictions = k_nearest_neighbors(X_train_edited, y_train_edited, X_test, k, regression=regression, sigma=sigma)

    # Calculate performance
    if regression:
        performance = np.mean((predictions - y_test) ** 2)
    else:
        performance = np.mean(predictions == y_test)

    return performance


def cross_validate_manual(X, y, k_values, num_folds=10, regression=False, sigma=None, error_threshold=None):
    """
    Perform manual 10-fold cross-validation with stratified splits and Edited k-NN, in parallel.
    """
    # Stratified k-fold split
    folds = stratified_k_fold_cross_validation(X, y, num_folds=num_folds, regression=regression)

    best_k = None
    best_sigma = None
    best_performance = float('inf') if regression else 0

    avg_performance_list = []
    whenK = []
    whenS = []
 
    rowNum = 0
    colNum = 0
    if sigma != None:
        
        for k in k_values:
            for s in sigma:
                # Prepare fold data for multiprocessing
                fold_data = []
                for i in range(num_folds):
                    test_set = folds[i]
                    train_set = np.concatenate(folds[:i] + folds[i+1:])

                    X_train, y_train = train_set[:, :-1], train_set[:, -1]
                    X_test, y_test = test_set[:, :-1], test_set[:, -1]

                    fold_data.append((X_train, y_train, X_test, y_test, int(k), regression, s, error_threshold))


                # Use multiprocessing to parallelize fold processing
                with mp.Pool(mp.cpu_count()) as pool:
                    performances = pool.map(process_fold, fold_data)

                avg_performance = np.mean(performances)
                avg_performance_list.append(float(avg_performance))
                whenK.append(k)
                whenS.append(s)
                print(f"Average performance for k={k} & s={s}: {avg_performance}")
                
                if (regression and avg_performance < best_performance) or (not regression and avg_performance > best_performance):
                    best_performance = avg_performance
                    best_k = int(k)
                    best_sigma = s

        whenS, newPerformList = zip(*sorted(zip(whenS, avg_performance_list)))
        plt.plot(whenK, avg_performance_list, '-o')
        plt.xticks(np.arange(1, 11, step=2))
        plt.show()
        plt.plot(whenS, newPerformList, '-o', color='green')
        plt.xticks(np.arange(.6, 1.6, step=.2))
        plt.show()
    else:
        for k in k_values:
                # Prepare fold data for multiprocessing
                fold_data = []
                for i in range(num_folds):
                    test_set = folds[i]
                    train_set = np.concatenate(folds[:i] + folds[i+1:])

                    X_train, y_train = train_set[:, :-1], train_set[:, -1]
                    X_test, y_test = test_set[:, :-1], test_set[:, -1]

                    fold_data.append((X_train, y_train, X_test, y_test, k, regression, None , error_threshold))

                # Use multiprocessing to parallelize fold processing
                with mp.Pool(mp.cpu_count()) as pool:
                    performances = pool.map(process_fold, fold_data)

                avg_performance = np.mean(performances)
                print(f"Average performance for k={k}: {avg_performance}")

                avg_performance_list.append(float(avg_performance))
                whenK.append(k)
                
                if (regression and avg_performance < best_performance) or (not regression and avg_performance > best_performance):
                    best_performance = avg_performance
                    best_k = int(k)
        plt.plot(whenK, avg_performance_list, '-o')
        plt.xticks(np.arange(1, 11, step=2))
        plt.show()

    # Ensure best_k is valid
    if best_k is None:
        raise ValueError("Failed to determine the best k. Check data preprocessing and model logic.")

    if regression:
        return best_k, best_sigma
    else:
        return best_k, best_sigma

