# linear-regression
📚 Problem
We want to predict house prices using input features like area, number of bedrooms, parking, etc.

🛠 Steps
1. Import Libraries
python
Copy
Edit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
Why?
We need tools to load data, split it, train a model, and evaluate results.

2. Load and Prepare Data
python
Copy
Edit
df = pd.read_csv('yourfile.csv')  # Example
X = df.drop('price', axis=1)
y = df['price']
Why?

X = Features (Inputs like area, bedrooms).

y = Target (Output: price).

3. Scale Features
python
Copy
Edit
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Why?
Scaling helps models like Linear Regression work better when features have different units (like meters vs number of rooms).

4. Train-Test Split
python
Copy
Edit
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
Why?
Split data into:

Training set → to learn

Testing set → to check performance

random_state=42 → makes results reproducible.

5. Train the Linear Regression Model
python
Copy
Edit
model = LinearRegression()
model.fit(X_train, y_train)
Why?
Fit the model to find the best line that predicts price.

6. Make Predictions
python
Copy
Edit
y_pred = model.predict(X_test)
Why?
Use the trained model to predict prices for new (test) data.

7. Evaluate the Model
python
Copy
Edit
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
Metrics:

MSE: Mean Squared Error (lower = better)

MAE: Mean Absolute Error (average error size)

R² Score: How much variation model explains (1 = perfect, closer to 1 is good)

8. Visualize Predictions
python
Copy
Edit
plt.scatter(y_test, y_pred, c='green')
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted values')
plt.show()
Why?
Visualize how close predicted prices are to actual prices.

🎯 Final Flow

Step	What Happened
Load Data	Read CSV
Split into X and y	Separate features and target
Scale Data	Standardize features
Train/Test Split	Divide into training and testing sets
Train Model	Fit Linear Regression
Predict	Predict on test data
Evaluate	Use MSE, MAE, R²
Visualize	Plot True vs Predicted
🧠 Tips for Beginners

Tip	Meaning
Always scale your features for models like Linear Regression.	
Always split data — never test on the training set!	
Lower MSE/MAE = better model. Higher R² (closer to 1) = better model.	
Visualize to understand errors easily.	
🚀 What Next?
Try using Polynomial Regression if results are not great.

Try Regularization (like Ridge or Lasso).

Tune hyperparameters like test size, feature scaling method.

Try different ML algorithms (RandomForestRegressor, etc.)

✅ Conclusion
You built, trained, evaluated, and visualized a complete Linear Regression model successfully! 🎯
This is the foundation of Machine Learning.
