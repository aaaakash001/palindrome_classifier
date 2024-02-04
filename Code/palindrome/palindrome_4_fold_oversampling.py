import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from imblearn.over_sampling import SMOTE

bit10_binary = pd.read_csv("Code/data/binary_strings_palindrome_check.csv", dtype={"X": object})

# Extract features (X) and labels (Y)
X = np.array([list(map(int, x)) for x in bit10_binary["X"]]).T
Y = np.array(bit10_binary["y"]).reshape((1, -1))

# Apply SMOTE to oversample the minority class
smote = SMOTE(sampling_strategy='minority', random_state=42)
X_resampled, Y_resampled = smote.fit_resample(X.T, Y.ravel())

X = X_resampled.T
Y = Y_resampled.reshape((1, -1))

print("shape of X:",X.shape)
print("shape of Y:",Y.shape)

print("Number of palindromes:",np.sum(Y))

def sigmoid(Z):
    Z = np.clip(Z, -500, 500)
    return 1 / (1 + np.exp(-Z))

def ReLU(Z):
    return np.maximum(0, Z)

def dReLU(Z):
    return Z > 0

def layer_sizes(X,H,Y):
    n_x = X.shape[0]
    n_h = H
    n_y = Y.shape[0]
    return n_x, n_h, n_y

def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(3)
    W1 = np.random.randn(n_h, n_x) *0.01
    # print("shape of W1:",W1.shape)
    b1 = np.zeros((n_h, 1))
    # print("shape of b1:",b1.shape)

    W2 = np.random.randn(n_y, n_h) *0.01
    # print("shape of W2:",W2.shape)

    b2 = np.zeros((n_y, 1))
    # print("shape of b2:",b2.shape)


    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters

def forward_propagation(X, parameters):
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    return A2, Z2, A1, Z1

def compute_cost(A2, Y):
    m = Y.shape[1]
    logprobs = Y * np.log(A2) + (1 - Y) * np.log(1 - A2)
    cost = -np.sum(logprobs) / m
    return cost

def backward_propagation(parameters, A1, A2, Z1, Z2, X, Y):
    m = X.shape[1]
    W2 = parameters["W2"]
    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.multiply(np.dot(W2.T, dZ2) ,(A1 * (1 - A1)))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads

def update_parameters(parameters, grads):
    W1, b1, W2, b2 = parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"]
    dW1, db1, dW2, db2 = grads["dW1"], grads["db1"], grads["dW2"], grads["db2"]

    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def predict(parameters, X):
    A2, _, _, _ = forward_propagation(X, parameters)
    predictions = (A2 >= 0.5).astype(int)
    return predictions, A2


def cross_validation(X, Y, n_h, n_folds, num_iterations, print_cost=False):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    accuracies = []
    precisions = []
    all_fold_parameters = []

    for fold, (train_index, test_index) in enumerate(kf.split(X.T), 1):
        X_train, X_test = X.T[train_index].T, X.T[test_index].T
        Y_train, Y_test = Y.T[train_index].T, Y.T[test_index].T

        n_x, n_h, n_y = layer_sizes(X_train, n_h, Y_train)
        parameters = initialize_parameters(n_x, n_h, n_y)

        for i in range(num_iterations):
            A2, Z2, A1, Z1 = forward_propagation(X_train, parameters)
            cost = compute_cost(A2, Y_train)
            grads = backward_propagation(parameters, A1, A2, Z1, Z2, X_train, Y_train)
            parameters = update_parameters(parameters, grads)

            if print_cost and i % 1000 == 0:
                print(f"Fold: {fold}, Cost after iteration {i}: {cost}")

        predictions_test, _ = predict(parameters, X_test)
        accuracy_test = np.mean(predictions_test == Y_test)
        precision_test = np.sum(np.logical_and(predictions_test == 1, Y_test == 1)) / np.sum(predictions_test == 1) if np.sum(predictions_test == 1) > 0 else 0
        print("Number of 1's predicted:",np.sum(predictions_test == 1),"Number of 1's in actual:",np.sum(Y_test == 1))
        accuracies.append(accuracy_test)
        precisions.append(precision_test)
        all_fold_parameters.append(parameters)
        print(f"Fold: {fold}, Accuracy: {accuracy_test * 100}%, Precision: {precision_test}")

    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    print(f"Average accuracy over {n_folds} folds: {mean_accuracy * 100}%")
    print(f"Average precision over {n_folds} folds: {mean_precision}")

    return mean_accuracy, all_fold_parameters, mean_precision


# Set n_h and num_iterations as needed
n_h = 5
num_iterations = 10000
learning_rate=1
n_folds=4
# Perform 4-fold cross-validation
mean_accuracy, all_fold_parameters,mean_precision = cross_validation(X, Y, n_h, n_folds,num_iterations, print_cost=True)

print(f"Average accuracy : {mean_accuracy * 100}% and precision: {mean_precision} over {n_folds} folds")

parameters_fold1 = all_fold_parameters[0]

# Generate a single example for prediction (you can replace this with your actual data)
example_to_predict = np.array([[1,1,1,1,1,1,1,1,1,1],[1,1,0,1,1,1,1,1,1,1],[0,1,1,1,1,1,1,1,1,0]]).T

# Make prediction using parameters from fold 1
prediction_fold1, A2 = predict(parameters_fold1, example_to_predict)
print("shape of A2:",A2.shape)
print("Example to predict:", example_to_predict)
print("A2 :", A2)
print("Prediction using fold 1 parameters:", prediction_fold1)