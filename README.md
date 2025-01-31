# ML-Assignment

We are importing the necessary libraries:
 pandas for data manipulation.
 train_test_split from sklearn to split the dataset.
 LinearRegression for creating the model.
 mean_squared_error for evaluating the model's performance.

data = pd.read_csv('housing_data.csv')
 This line reads the CSV file (housing_data.csv) into a pandas DataFrame called data. You need to replace 'housing_data.csv' with the actual path to your dataset file.
X = data[['GrLivArea', 'YearBuilt']]
y = data['SalePrice']
 X contains the features (input variables) we want to use for prediction. Here, we're using the size of the house (GrLivArea) and the year it was built (YearBuilt).
 y is the target variable (house prices) we want to predict.

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  this line splits the dataset into training and testing sets:
  80% of the data for training (X_train, y_train).
  20% for testing (X_test, y_test). The random_state=42 ensures the split is reproducible.

model = LinearRegression()
model.fit(X_train, y_train)
 This initializes a linear regression model.
 The model.fit(X_train, y_train) line trains the model using the training data (X_train, y_train).

y_pred = model.predict(X_test)
  This line uses the trained model to make predictions on the test data (X_test).

mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
  This calculates the Mean Squared Error (MSE), which measures the average of the squared differences between actual and predicted values. Lower values indicate 
  better performance.
  The result is printed to the console.
