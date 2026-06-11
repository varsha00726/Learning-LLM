import pandas as pd
from sklearn.linear_model import LinearRegression

# Step 1: Load dataset
data = pd.read_csv("salary.csv")

# Step 2: Split input/output
X = data[['Experience']]
y = data['Salary']

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict
prediction = model.predict([[6]])
print(f"Predicted salary for 6 years experience: {prediction[0]}")