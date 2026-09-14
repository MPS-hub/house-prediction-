import pandas as pd
from sklearn.linear_model import LinearRegression
df = pd.read_csv("House Price Prediction Dataset.csv")
print(df.head())
X = df[["Area"]]
y = df["Price"]
model = LinearRegression()
model.fit(X, y)
area = float(input("Enter house area in sq ft: "))
prediction = model.predict([[area]])
print("Predicted House Price:", prediction[0])