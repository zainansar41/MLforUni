import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

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

onehot_encoder = OneHotEncoder(drop='first', sparse_output=False)

encoded_data = onehot_encoder.fit_transform(file[categorical_columns])

feature_names = onehot_encoder.get_feature_names_out(categorical_columns)

encoded_df = pd.DataFrame(encoded_data, columns=feature_names)

file.drop(categorical_columns, axis=1, inplace=True)

file = pd.concat([file, encoded_df], axis=1)

# Split the data into features and target variable
X = file.drop('price', axis=1).to_numpy()  # Convert X to a NumPy array
y = file['price'].to_numpy()  # Convert y to a NumPy array

# Initialize model parameters
model = LinearRegression()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit the Linear Regression model
model.fit(X_train, y_train)

# Now you can use the trained model for predictions and evaluation

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model

# Calculate the Mean Squared Error
mse = np.mean((y_pred - y_test) ** 2)
print("Mean Squared Error:", mse)


