import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("House Price Prediction Dataset.csv")

print("--- First 5 rows ---")
print(df.head())

print("\n--- Dataset Shape ---")
print(df.shape)

print("\n--- Missing Values ---")
print(df.isna().sum())

print("\n--- Dataset Info ---")
df.info()

X = df[['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n--- Model Evaluation ---")
print("MSE:", mse)
print("R2 Score:", r2)

print("\n--- Correlations with Price ---")
print("Id vs Price:", df['Id'].corr(df['Price']))
print("Area vs Price:", df['Area'].corr(df['Price']))

print("\n--- Correlation Matrix ---")
print(df.corr(numeric_only=True))