import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt



df = pd.read_csv("ev_battery_dataset.csv")

X = df.drop("Battery_Degradation_%",axis = 1)
Y = df["Battery_Degradation_%"]

print(X.head())
print(Y.head())

#splittin the data into training and testing

X_train,X_test,Y_train,Y_test = train_test_split(
    X,Y,test_size=0.2,random_state=42
)

print("X_train : ",X_train.shape)
print("Y_train : ",Y_train.shape)
print("X_test : ",X_test.shape)
print("Y_test : ",Y_test.shape)

#creating model

model = LinearRegression()

#training model

model.fit(X_train,Y_train)

print("model trained succesfully!")

#make predictions

Y_pred = model.predict(X_test)

print(Y_pred[:5])

#evaluating the model

from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("MAE:", mae)
print("R2 Score:", r2)

print("Coefficients:")
print(model.coef_)

print("\nIntercept:")
print(model.intercept_)

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print(feature_importance)


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, Y_train)

rf_pred = rf_model.predict(X_test)

rf_mae = mean_absolute_error(Y_test, rf_pred)
rf_r2 = r2_score(Y_test, rf_pred)

print("Random Forest MAE:", rf_mae)
print("Random Forest R2:", rf_r2)

#Feature importance


importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance)



plt.bar(
    importance["Feature"],
    importance["Importance"]
)

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()

#show the model

import joblib

joblib.dump(
    rf_model,
    "ev_battery_model.pkl"
)

print("Model Saved")

sample = pd.DataFrame({
    "Charge_Cycles": [1500],
    "Fast_Charging_Frequency_%": [40],
    "Avg_Temperature_C": [35],
    "Driving_Aggression_Index": [0.7]
})

prediction = rf_model.predict(sample)

print(prediction)

battery_health = 100 - prediction[0]

print("Battery Health:", battery_health)
