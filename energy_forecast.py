
# World Energy Consumption Forecasting using LSTM

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense


# Step 2: Load and preprocess the dataset
df = pd.read_csv("World_Energy_By_Country_And_Region_1965_to_2023.csv")

# Select India
india = df[df["Country"] == "India"].copy()

# Convert wide format to yearly format
india_yearly = india.melt(
    id_vars=["Country"],
    var_name="Year",
    value_name="Energy_Consumption"
)

# Convert columns to numeric types
india_yearly["Year"] = india_yearly["Year"].astype(int)

india_yearly["Energy_Consumption"] = pd.to_numeric(
    india_yearly["Energy_Consumption"],
    errors="coerce"
)

# Handle missing values
india_yearly["Energy_Consumption"] = (
    india_yearly["Energy_Consumption"].interpolate()
)

india_yearly["Energy_Consumption"] = (
    india_yearly["Energy_Consumption"].ffill().bfill()
)

# Sort data chronologically
india_yearly = (
    india_yearly.sort_values("Year").reset_index(drop=True)
)

print("Dataset loaded successfully!")
print("Total yearly records:", len(india_yearly))


# Step 3: Normalize data and create sequences
scaler = MinMaxScaler()

india_yearly["Energy_Normalized"] = scaler.fit_transform(
    india_yearly[["Energy_Consumption"]]
)

values = india_yearly["Energy_Normalized"].values

X = []
y = []

# Use previous 5 years to predict the next year
for i in range(len(values) - 5):
    X.append(values[i:i + 5])
    y.append(values[i + 5])

X = np.array(X)
y = np.array(y)

print("Input sequences shape:", X.shape)
print("Target values shape:", y.shape)


# Step 4: Chronological split and train the LSTM model
split_index = int(len(X) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Reshape input for LSTM: samples, time steps, features
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# Build the LSTM model
model = Sequential([
    Input(shape=(5, 1)),
    LSTM(32),
    Dense(1)
])

model.compile(
    optimizer="adam",
    loss="mse"
)

model.summary()

# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    verbose=1
)

print("Model training completed!")


# Predict test values
y_pred = model.predict(X_test, verbose=0).flatten()

print("\nPredicted normalized values:")
print(y_pred)

print("\nActual normalized values:")
print(y_test)


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\nModel Evaluation")
print("MAE:", mae)
print("RMSE:", rmse)


# Step 5: Visualize actual vs predicted values
plt.figure(figsize=(10, 5))

plt.plot(
    y_test,
    label="Actual Energy Consumption",
    marker="o"
)

plt.plot(
    y_pred,
    label="Predicted Energy Consumption",
    marker="x"
)

plt.title("India Energy Consumption: Actual vs Predicted")
plt.xlabel("Test Sample")
plt.ylabel("Normalized Energy Consumption")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()
