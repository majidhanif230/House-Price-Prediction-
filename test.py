# model.py

import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import numpy as np
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("Housing.csv")

# Print column names to verify correctness
print("Columns in the DataFrame:", df.columns)

# Drop 'Address' column if it exists
if 'Address' in df.columns:
    df = df.drop('Address', axis=1)

# Visualize relationships between variables
sns.pairplot(df)
plt.show()

# Check for missing values
if df.isnull().sum().any():
    df = df.dropna()

# Define the columns to log transform
features_to_log = ['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']

# Verify that the columns exist before applying log transformation
missing_columns = [col for col in features_to_log if col not in df.columns]
if missing_columns:
    raise KeyError(f"The following columns are missing from the DataFrame: {missing_columns}")

# Log transform features
df[features_to_log] = df[features_to_log].apply(np.log)

# Split features and target variable
X = df.drop('Price', axis=1)
y = df['Price']

# Standardize features
sc = StandardScaler()
X = pd.DataFrame(sc.fit_transform(X), columns=X.columns)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Check model performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print("Train Score:", train_score)
print("Test Score:", test_score)

# Save the model to disk
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the scaler to disk
with open("scaler.pkl", "wb") as file:
    pickle.dump(sc, file)

# Load the model from disk
with open("model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Check loaded model performance
result = loaded_model.score(X_test, y_test)
print("Loaded Model Test Score:", result)
