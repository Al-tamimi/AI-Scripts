import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load stock market data into a Pandas DataFrame
data = pd.read_csv('stock_prices.csv')

# Drop any rows with missing values
data = data.dropna()

# Split the data into training and testing sets
X = data.drop(['Date', 'Price'], axis=1)
y = data['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a linear regression model on the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model on the testing data
score = model.score(X_test, y_test)
print("Model score:", score)

# Predict the stock price for the next day
last_day = data.iloc[-1].drop(['Date', 'Price']).values.reshape(1, -1)
predicted_price = model.predict(last_day)
print("Predicted price:", predicted_price)
