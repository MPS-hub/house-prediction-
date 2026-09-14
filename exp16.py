import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import MinMaxScaler as mm, LabelEncoder

path = "House Price Prediction Dataset - House Price Prediction Dataset.csv"
df = pd.read_csv(path)

df.drop_duplicates(inplace=True)

le = LabelEncoder()
df["Location"] = le.fit_transform(df["Location"])
df["Condition"] = le.fit_transform(df["Condition"])
df["Garage"] = le.fit_transform(df["Garage"])

df.drop(["Id"], axis=1, inplace=True)

df.to_csv(path, index=False)

x = df.drop(["Price"], axis=1)
y = df["Price"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

sc = mm()
x_train_scaled = sc.fit_transform(x_train)
x_test_scaled = sc.transform(x_test)

model = LinearRegression()
model.fit(x_train_scaled, y_train)

y_pred = model.predict(x_test_scaled)
score = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("R2 Score:", score)
print("Mean Squared Error (MSE):", mse)
print(df.corr(numeric_only=True))
