import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
import copy
import joblib
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load the housing dataset
file = pd.read_csv("D:\Github\MLforUni\Housing.csv")

# Preprocess the dataset
file.dropna(inplace=True)

# Create a list of categorical columns to one-hot encode
categorical_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea",
    "furnishingstatus",
]

onehot_encoder = OneHotEncoder(drop='first', sparse=False)

encoded_data = onehot_encoder.fit_transform(file[categorical_columns])

feature_names = onehot_encoder.get_feature_names_out(categorical_columns)

encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

file.drop(categorical_columns, axis=1, inplace=True)

file = pd.concat([file, encoded_df], axis=1)

# Normalize the 'price' and 'area' columns using Min-Max scaling
min_max_scaler = MinMaxScaler()
file['price'] = min_max_scaler.fit_transform(file['price'].values.reshape(-1, 1))
file['area'] = min_max_scaler.fit_transform(file['area'].values.reshape(-1, 1))

# Split the data into features and target variable
X = file.drop('price', axis=1).to_numpy()  # Convert X to a NumPy array
y = file['price'].to_numpy()  # Convert y to a NumPy array

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize model parameters
w_initial = np.full(X_train.shape[1], 0.5)
b_initial = 0

# Define the cost function (Mean Squared Error)
def compute_cost(X, y, w, b):
    m = X.shape[0]
    predictions = np.dot(X, w) + b
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Define the gradient function
def compute_gradient(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):
        err = np.dot(X[i], w) + b - y[i]
        dj_db = dj_db + err
        dj_dw = dj_dw + err * X[i]

    dj_dw = dj_dw / m
    dj_db = dj_db / m

    return dj_db, dj_dw

# Gradient Descent
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, threshold=0.00001):
    J_history = []
    w = copy.deepcopy(w_in)
    b = b_in

    for i in range(num_iters):
        dj_db, dj_dw = gradient_function(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db

        cost = cost_function(X, y, w, b)
        J_history.append(cost)

        if i % 100 == 0:
            print(f"Iteration {i}: Cost {cost}")

        # Check for convergence
        if i > 0 and abs(J_history[i - 1] - J_history[i]) < threshold:
            break

    return w, b, J_history

# Reduce the learning rate
alpha = 0.01

# Number of iterations
num_iters = 100000

w_final, b_final, J_history = gradient_descent(X_train, y_train, w_initial, b_initial, compute_cost, compute_gradient, alpha, num_iters)

# # Save the trained model to a file
# model_filename = "linear_regression_model.pkl"
# joblib.dump((w_final, b_final, min_max_scaler, onehot_encoder), model_filename)

# # Plot the cost history
# plt.plot(J_history)
# plt.xlabel("Iterations")
# plt.ylabel("Cost")
# plt.title("Cost History")
# plt.show()
