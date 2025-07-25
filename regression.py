import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load CSV
file_path = r"D:\ML Data\housrent prediction\house_data.csv"
data = pd.read_csv(file_path)

# Remove extra spaces from column names
data.columns = data.columns.str.strip()

# Inputs and Output
X = data[['area', 'bedrooms', 'bathrooms']]
y = data['price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluation
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# Predict a new house
new_house = [[2000, 3, 2]]
predicted_price = model.predict(new_house)
print("Predicted price for new house:", predicted_price[0])

# ------------------------------
# 📊 1. Actual vs Predicted Prices
# ------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y), max(y)], [min(y), max(y)], color='red', linestyle='--')  # reference line
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted House Prices")
plt.grid(True)
plt.show()

# ------------------------------
# 📈 2. Area vs Price
# ------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(data['area'], data['price'], color='green')
plt.xlabel("Area (sq ft)")
plt.ylabel("Price")
plt.title("Area vs Price")
plt.grid(True)
plt.show()
